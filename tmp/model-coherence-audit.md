# Local model coherence audit (rescored)

Prompt: `Answer with exactly one short English sentence. What is the capital of France?`

Scope: **29 local completion models** (skipped ~400 cloud tags, embeddings, lab stubs).

## Verdict summary

| Verdict | Count | Meaning |
|---|---:|---|
| **ok** | **19** | Ran and answered with Paris (or coherent short Paris) |
| **weak / cut off** | **0** after retest | qwen3.6:27b twins OK with longer `num_predict` |
| **error (env)** | **7** | Need `llama-server` for draft/ngram/mtp/dflash |
| **error (panic)** | **1** | `gemma4:26b-optiq` MLX panic |

## OK (coherent)

- `lfm2-350m-mlx:4bit` (~551 tok/s) — Paris
- `lfm2-350m:q4_k_m` (~418)
- `qwen2.5-0.5b-mlx:4bit` (~366) — OK but continues past `<|endoftext|>` (stop-token quirk)
- `qwen2.5:0.5b`, `qwen2.5:1.5b`, `qwen3:0.6b`
- `driaforall/tiny-agent-a-0.5b` (:latest / :q4_k / :q8_0) + `mradermacher/...:f16`
- `m21-ggml:latest`, `llama3.2:1b`, `llama3.2:3b`
- `bonsai:27b`, `bonsai:27b-mlx`, `ornith-9b-optiq`
- `mlx-community/gemma-3-27b-it-qat-4bit`
- `lmstudio-community/hermes-4-70b` (~6 tok/s)
- `lmstudio-community/qwen3-coder-next` (~55 tok/s @ ctx 1024)
- `qwen3.6:27b`, `lmstudio-community/qwen3.6-27b:q8_0` (thinking models; need ≥~128 predict tokens)

## Failures

### Real runner bug
- **`gemma4:26b-optiq`** — `mlx runner` panic (`index out of range [0]`) — same class as pre-fix bonsai/ornith; weights path still broken for this OptIQ gemma4.

### Missing llama-server (not gibberish)
These refuse load until draft server is built/configured:
- `eliza-1-2b`, `eliza-1-2b-ngram`
- `eliza-1-27b-256k`, `-ngram`, `-dflash`
- `qwen3.6:latest`, `qwen3.6-mtp:latest`

Error pattern: `model requires llama-server for draft-* / ngram-* / dflash`.

## Notes
- No true “random unicode gibberish” on models that successfully decoded.
- Tiny MLX agents sometimes emit repeated `<|endoftext|>` — stop handling, not weight corruption.
- Raw results: `/tmp/model-coherence-audit.jsonl`
