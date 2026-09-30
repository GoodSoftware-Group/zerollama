---
name: multidecode
description: "Shared-prefix forest decode via zerollama POST /v1/multidecode — custom harness packs tokens/pos/parent (or sticky parent_node_ids); not chat, not Hermes batch (that path may accelerate silently)."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, multidecode, forest, llama-cpp, batch, harness]
    category: mlops
    related_skills: [zerollama-integration, batch-inference, hermes-provider]
---

# MultiDecode Skill

Run **shared-prefix forest decode** on
[zerollama](https://github.com/GoodSoftware-Group/zerollama) via
`POST /v1/multidecode`.

**Why not chat:** one forward scores many branch leaves with a shared KV prefix
(custom RoPE `pos` + ancestor mask). Vanilla `/v1/chat/completions` stays
single-stream.

**Why not Hermes batch:** Hermes keeps `POST /v1/chat/completions/batch`; the
server may silently pack when LCP ≥ 32. Custom harnesses that already own
token ids should call `/v1/multidecode` directly.

Docs: [docs/multidecode-llama-cpp.md](../../docs/multidecode-llama-cpp.md) ·
[findings](../../docs/multidecode-llama-cpp-findings.md) · OpenAPI
`MultiDecodeRequest`.

## Compatibility check

```bash
# Lab ports only — never bind :11434 / :8081 from agent work
curl -sS http://127.0.0.1:11435/api/version | python3 -c \
  'import sys,json; c=json.load(sys.stdin).get("zerollama",{}).get("capabilities",{}); print(c.get("multidecode"))'
# expect: True
```

`null` / missing → build predates MD1 (patches **0129–0131**).

## When to Use

- Multi-question / multi-trace fan-out with a **long shared prefix**
- Beam-like or parallel continuations that should share one prefix KV
- You already tokenize and can build `tokens` / `pos` / `parent` / `leaves`

## When NOT to Use

- Open-ended single chat → `/v1/chat/completions` or `/api/chat`
- OpenAI-shaped batch without owning tokens → `/v1/chat/completions/batch`
- SWA / sliding-window models (not supported)
- Forests larger than one ubatch (`n_batch` / `n_ubatch`)

## Prerequisites

- Patches **0129–0131** + llama-server with `--multidecode` (Go non-embedding
  launches pass this automatically)
- Local GGUF model tag (not `:cloud`)
- Lab bind e.g. `:11435` / llama-server `:18082`

## API contract

`POST /v1/multidecode`

| Field | Required | Notes |
|-------|----------|-------|
| `model` | yes (zerollama) | Local tag; omitted on raw llama-server |
| `tokens` | yes | Token ids for this decode step |
| `pos` | yes | Tree depth (RoPE); siblings may share a depth |
| `parent` | one-shot | Batch index of parent, or `-1` root; parents earlier in batch |
| `parent_node_ids` | multi-step | Durable KV parents; omit `parent` when set |
| `leaves` | yes | Indices into `tokens` that return logits / greedy token |
| `clear` | no | Default `true` when `parent_node_ids` omitted; use `false` for sticky |
| `return_logits` | no | Full vocab rows (large); default greedy `token` only |
| `node_ids` | no | Client-assigned durable ids; else server allocates |

**Response:** `{ "results":[{"leaf","token","node_id","logits?"}], "node_ids":[...] }`

**Multi-step:** client loops. `n_predict` is accepted but the server does **not**
autoregressive-loop — each sticky step is a new POST with `clear:false`.

## Pack a forest (shared prefix)

Given tokenized sequences that share a prefix of length `L`:

1. Emit prefix once: `pos = 0..L-1`, `parent = [-1,0,1,…]`
2. For each sequence, append its unique suffix as a branch from the prefix leaf
3. `leaves[i]` = batch index of the last token of branch `i`

```
prefix:  t0 → t1 → t2
                  ↘ a3 → a4     leaf A
                  ↘ b3 → b4 → b5 leaf B
```

Go helper: `llm.PackForestFromTokenLists` (same rules as silent Hermes pack).

## How to run (zerollama lab)

```bash
BASE=http://127.0.0.1:11435
MODEL=llama3.2:3b   # local tag

# 1) Tokenize (via llama-server lab, or your matching tokenizer)
#    Example linear chain already tokenized:
#    tokens=[791,6864,315,9822,374]  # "The capital of France is"

# 2) One-shot forest (here: single linear path — trivial forest)
curl -sS "$BASE/v1/multidecode" -H 'content-type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"tokens\": [791,6864,315,9822,374],
  \"pos\": [0,1,2,3,4],
  \"parent\": [-1,0,1,2,3],
  \"leaves\": [4],
  \"clear\": true
}"
# → results[0].token (greedy next), results[0].node_id, node_ids=[0,1,2,3,4]

# 3) Sticky step: write previous greedy token under leaf node_id, predict next
#    (keep the same runner warm — keep_alive / same process)
LEAF_NID=4   # from results[0].node_id
PREV=279     # from results[0].token
curl -sS "$BASE/v1/multidecode" -H 'content-type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"tokens\": [$PREV],
  \"pos\": [5],
  \"parent_node_ids\": [$LEAF_NID],
  \"leaves\": [0],
  \"clear\": false
}"
# → new node_ids for written tokens; results[0].token is the next prediction
```

### Two-branch one-shot (sketch)

Shared prefix ids `P`, then branch A suffix / branch B suffix:

```json
{
  "model": "llama3.2:3b",
  "tokens": ["…P…", "…A…", "…B…"],
  "pos": ["…", "…", "…"],
  "parent": ["…", "…", "…"],
  "leaves": [leafA, leafB],
  "clear": true
}
```

Use `PackForestFromTokenLists([[…P…A…],[…P…B…]])` rather than hand-building.

## Raw llama-server (optional)

```bash
./build/bin/llama-server -m MODEL.gguf --multidecode -ngl 0 \
  --port 18082 --host 127.0.0.1
# Same JSON without "model"
```

## Pitfalls

- **Sibling leak without MultiDecode:** same `pos` + causal mask lets later leaves see earlier siblings — always set `parent` / sticky `parent_node_ids`.
- **Tokenizer mismatch:** ids must match the loaded GGUF vocab.
- **Ubatch size:** whole forest in one decode; raise `-b` / `-ub` if needed.
- **Production ports:** do not point agent smokes at `:11434` / `:8081`.
