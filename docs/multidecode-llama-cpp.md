# MultiDecode (llama.cpp) — operator notes

**Patches:**
- [`0129`](../llama/patches/0129-llama-multidecode-ancestor-kq-mask.patch) — `batch.parent` ancestor KQ mask (MD0)
- [`0130`](../llama/patches/0130-llama-multidecode-cell-node-id.patch) — cell `node_id` / `parent_node_id` + multi-step mask (MD1a)
- [`0131`](../llama/patches/0131-server-multidecode-POST-v1-multidecode.patch) — llama-server `--multidecode` + `POST /v1/multidecode` (MD1b)

**Upstream idea:** [WestCoastML/multidecode](https://github.com/WestCoastML/multidecode)  
**Findings:** [multidecode-llama-cpp-findings.md](./multidecode-llama-cpp-findings.md)  
**Pin:** [runtime/LLAMA_CPP_PIN.md](../runtime/LLAMA_CPP_PIN.md)  
**Hermes batch:** [hermes-zerollama-gap.md](./hermes-zerollama-gap.md) §8

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
| `node_id` / `parent_node_id` | Durable ids written into KV cells for multi-step (0130) |
| Sequences | Single `seq_id` for the whole forest |
| Ubatch | Entire forest must fit in one ubatch (`n_batch` / `n_ubatch` ≥ `n_tokens`) |
| SWA | Not supported (assert) |

`llama_batch_init` leaves `parent` / `node_id` / `parent_node_id` NULL.

## HTTP (llama-server + zerollama)

Gate: `--multidecode` (Go llama-server launches pass this for non-embedding models).

**One-shot:**

```json
{"tokens":[...],"pos":[...],"parent":[...],"leaves":[4,6],"return_logits":false}
```

→ `{"results":[{"leaf":4,"token":…,"node_id":4}],"node_ids":[0,1,…]}`

**Multi-step (sticky KV, same server slot):**

```json
{"tokens":[30,31],"pos":[5,5],"parent_node_ids":[4,6],"leaves":[0,1],"clear":false}
```

Server allocates new `node_ids`, writes cells, returns them. Omit `parent` when `parent_node_ids` is set.

**zerollama:** `POST /v1/multidecode` (same body + `model`) → schedules llama-server runner. Capability: `GET /api/version` → `zerollama.capabilities.multidecode`.

Lab bind e.g. `:18082` — never `:11434` / `:8081`.

```bash
# llama-server lab
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
| **Custom harness** | `POST /v1/multidecode` — `tokens` + `pos` + `parent` / `parent_node_ids` + `leaves` |
| **Hermes / ElizaOS** | Unchanged `POST /v1/chat/completions/batch`; server may accelerate when prefixes match |
| **Vanilla chat** | Unchanged — do not stuff `parent[]` into normal `/v1/chat/completions` |

## Non-goals

- Hermes / ElizaOS client PRs
- ctypes `parent` / in-process Python forest
- SWA / multi-stream forests
- Production binds on `:11434` / `:8081`
