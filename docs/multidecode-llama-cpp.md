# MultiDecode (llama.cpp) — operator notes

**Patches** (b10969 series; numbers shifted after obsolete grammar patch drop):
- [`0126`](../llama/patches/0126-llama-MultiDecode-ancestor-KQ-mask-via-batch.parent-.patch) — `batch.parent` ancestor KQ mask (MD0)
- [`0127`](../llama/patches/0127-llama-MultiDecode-cell-node_id-multi-step-ancestor-m.patch) — cell `node_id` / `parent_node_id` + multi-step mask (MD1a)
- [`0128`](../llama/patches/0128-server-multidecode-and-POST-v1-multidecode-0131.patch) — llama-server `--multidecode` + `POST /v1/multidecode` (MD1b)

**Upstream idea:** [WestCoastML/multidecode](https://github.com/WestCoastML/multidecode)  
**Findings:** [multidecode-llama-cpp-findings.md](./multidecode-llama-cpp-findings.md)  
**Pin:** [runtime/LLAMA_CPP_PIN.md](../runtime/LLAMA_CPP_PIN.md)  
**Hermes batch:** [hermes-zerollama-gap.md](./hermes-zerollama-gap.md) §8  
**Harness skill:** [`agentskills/multidecode/`](../agentskills/multidecode/) (mirror [`skills/multidecode/`](../skills/multidecode/))

## Why this exists

Autoregressive decode usually emits one next-token prediction per forward (the last position). Self-attention can score many positions in parallel. MultiDecode packs a **forest** of sequences that share a prefix into one batch: custom RoPE `pos` = tree depth, and an **ancestor-only** KQ mask so siblings do not attend each other while sharing one physical KV for the common prefix.

**Why not dflash / EAGLE:** those are draft→verify speculative decode along one stream. MultiDecode is exact fan-out across independent branches (multi-question, beam, parallel traces).

**Why not only multi-seq + `seq_cp`:** that duplicates prefix KV per branch. MultiDecode keeps the shared prefix once.

## Lib API

```c
typedef struct llama_batch {
    // ... existing fields ...
    int32_t * parent;           // [n_tokens]; NULL = off (standard causal)
    int32_t * node_id;          // [n_tokens]; NULL = synth 0..n-1 (MD0 one-shot)
    int32_t * parent_node_id;   // [n_tokens]; NULL = derive from parent[] + node_id
} llama_batch;
```

| Rule | Detail |
|------|--------|
| `parent[i]` | Batch index of parent, or `-1` for a root; parents must appear earlier in the batch |
| `pos[i]` | Tree depth for RoPE (siblings may share the same depth) |
| `node_id` / `parent_node_id` | Durable ids written into KV cells for multi-step (0127) |
| Sequences | Single `seq_id` for the whole forest |
| Ubatch | Entire forest must fit in one ubatch (`n_batch` / `n_ubatch` ≥ `n_tokens`) |
| SWA | Not supported (assert) |

`llama_batch_init` leaves `parent` / `node_id` / `parent_node_id` NULL.

## Custom harness (zerollama)

End-to-end for harnesses that own tokenization. Lab ports only (`:11435`, llama-server `:18082`) — never `:11434` / `:8081`.

### 1. Probe

```bash
curl -sS http://127.0.0.1:11435/api/version | python3 -c \
  'import sys,json; print(json.load(sys.stdin)["zerollama"]["capabilities"].get("multidecode"))'
# True
```

### 2. Tokenize

Ids must match the loaded GGUF. Options:

- Lab `llama-server` `POST /tokenize` (`{"content":"…","add_special":false}`)
- Your client tokenizer for the same vocab
- Chat templates: render the full prompt string first, then tokenize (same as chat)

### 3. Pack the forest

Shared prefix of length `L`, then per-branch suffixes:

```
prefix:  t0 → t1 → t2
                  ↘ a3 → a4        leaf A
                  ↘ b3 → b4 → b5   leaf B
```

| Array | Rule |
|-------|------|
| `tokens` | Prefix once, then each unique suffix in batch order |
| `pos` | Tree depth (`0..L-1` for prefix; continue per branch) |
| `parent` | Batch index of parent, or `-1` for root; **parents must appear earlier** |
| `leaves` | Batch index of the last token of each branch |

Identical prompts share one leaf. Prefer `llm.PackForestFromTokenLists` (Go) over hand-building — same helper as silent Hermes pack (LCP ≥ 32).

### 4. One-shot via zerollama

```bash
BASE=http://127.0.0.1:11435
MODEL=llama3.2:3b

curl -sS "$BASE/v1/multidecode" -H 'content-type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"tokens\": [791,6864,315,9822,374],
  \"pos\": [0,1,2,3,4],
  \"parent\": [-1,0,1,2,3],
  \"leaves\": [4],
  \"clear\": true
}"
```

Response shape:

```json
{
  "results": [{"leaf": 4, "token": 279, "node_id": 4}],
  "node_ids": [0, 1, 2, 3, 4]
}
```

- `results[].token` — greedy argmax (set `"return_logits": true` for full rows)
- `results[].node_id` — durable id of that leaf in KV (parent for the next sticky write)
- `node_ids` — id assigned to each token in this request

### 5. Sticky multi-step (same runner)

The server does **not** loop on `n_predict`. After step 1, write the sampled token under the leaf’s `node_id`, predict the next:

```bash
# PREV = results[0].token, LEAF_NID = results[0].node_id, pos = prompt_len
curl -sS "$BASE/v1/multidecode" -H 'content-type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"tokens\": [279],
  \"pos\": [5],
  \"parent_node_ids\": [4],
  \"leaves\": [0],
  \"clear\": false
}"
```

Omit `parent` when `parent_node_ids` is set. Keep the runner warm (`keep_alive`) so KV sticks. Repeat: send last sampled token, `parent_node_ids` = previous response `results[0].node_id` (id of the token just written), `clear:false`.

### Two-branch sketch

```json
{
  "model": "llama3.2:3b",
  "tokens": [10, 11, 12, 20, 30, 31],
  "pos":    [0,  1,  2,  3,  3,  4],
  "parent": [-1, 0,  1,  2,  2,  4],
  "leaves": [3, 5],
  "clear": true
}
```

Prefix `[10,11,12]`; branch A `[20]`; branch B `[30,31]`. Leaves at batch indices 3 and 5.

## HTTP (llama-server direct)

Gate: `--multidecode` (Go llama-server launches pass this for non-embedding models).

Same wire **without** `model`:

```bash
./build/bin/llama-server -m MODEL.gguf --multidecode -ngl 0 --port 18082 --host 127.0.0.1

curl -sS http://127.0.0.1:18082/v1/multidecode -H 'content-type: application/json' -d '{
  "tokens":[791,6864,315,9822,374],"pos":[0,1,2,3,4],
  "parent":[-1,0,1,2,3],"leaves":[4],"clear":true
}'
```

## Hermes batch (silent pack)

`POST /v1/chat/completions/batch` — if `capabilities.multidecode` and tokenized prompts share LCP ≥ 32, Go packs a forest via `llm.PackForestFromTokenLists`, runs MultiDecode (+ sticky steps for `max_tokens`), maps into `chat.completion.batch`. Else: existing Python `generate_batch`. **No Hermes client PR.**

## Lab exactness

```bash
cmake -S vendor/llama-cpp-b10615 -B /tmp/md-build -DGGML_CUDA=OFF -DLLAMA_BUILD_EXAMPLES=ON
cmake --build /tmp/md-build --target llama-multidecode-exactness -j"$(nproc)"

MULTIDECODE_MODEL=/path/to/model.gguf \
  /tmp/md-build/bin/llama-multidecode-exactness -ngl 0
# no model → exit 0 (CI skip)
```

Step1 forest + step2 sticky append; leaf logits vs linear path must match (diff 0).

## Dual clients

| Client | Wire |
|--------|------|
| **Custom harness** | `POST /v1/multidecode` — pack + one-shot / sticky steps (skill above) |
| **Hermes / ElizaOS** | Unchanged `POST /v1/chat/completions/batch`; server may accelerate when prefixes match |
| **Vanilla chat** | Unchanged — do not stuff `parent[]` into normal `/v1/chat/completions` |

## Non-goals

- Hermes / ElizaOS client PRs
- ctypes `parent` / in-process Python forest
- SWA / multi-stream forests
- Production binds on `:11434` / `:8081`
