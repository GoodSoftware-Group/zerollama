# Speculative decoding (Phase 5)

One active method per model, implemented via **llama-server** `--spec-type` flags (pinned llama.cpp).

## Config (`speculative` block)

| `method` | llama.cpp type | Needs draft GGUF |
|----------|----------------|------------------|
| `none` | — | no |
| `ngram` | `ngram-simple` | no |
| `draft` / `draft-simple` | `draft-simple` | yes |
| `eagle3` / `dflash` | `draft-eagle3` | yes |
| `mtp` | `draft-mtp` | yes (often bundled) |

Example: [`configs/dual_4090_ngram.yaml`](../configs/dual_4090_ngram.yaml)

```yaml
speculative:
  method: ngram
  ngram:
    size_n: 12
    size_m: 48
    min_hits: 1
```

Draft model (second GGUF; use tensor split / TP in parent config for dual-GPU):

```yaml
speculative:
  method: draft-simple
  draft_model: /path/to/draft.gguf
  draft:
    n_max: 16
    n_gpu_layers: -1
```

## Environment

| Variable | Purpose |
|----------|---------|
| `ZEROLLAMA_SPEC_METHOD` | Override `speculative.method` |
| `LLAMA_DRAFT_MODEL` | Override draft GGUF path |

## Benchmark

Compare tokens/s with and without spec on the same prompt batch; use `GET /health` on the runtime for slot and pool utilization.
