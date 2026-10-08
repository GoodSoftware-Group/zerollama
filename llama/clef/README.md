# Clef joint head (L6)

Apache-2.0 joint schema head used by Ollama’s System One **Clef** path.
Ported from Cloudflare’s reference (`joint_schema_model.py`); refreshed from
`../ollama-upstream/llama/clef/` via `./scripts/phase/stage_clef_for_pin.sh`.

| Piece | Role |
|-------|------|
| `clef.h` / `clef.cpp` | CPU ggml joint head — consumes hidden states + `score_fields` spans |
| `upstream-002-clef.patch` | Mirror of `llama/compat/002-clef.patch` (server + context + CMake wire) |
| `UPSTREAM_SOURCE.txt` | Refresh provenance |

## Status on **b11351** (current pin)

| Layer | State |
|-------|--------|
| Head object-compile | Linked via `ZEROLLAMA_CLEF_DIR` into `server-context` |
| Server wire | In `llama/patches/0135-*` + `llama/compat/002-clef.patch` (`already_applied` on vendor) |
| CMake link | `build_llama_server.sh` passes `-DZEROLLAMA_CLEF_DIR=…/llama/clef` |
| Live GGUF score | Needs a Clef GGUF + `--embeddings`; chat smoke OK with head linked |

```bash
./scripts/phase/stage_clef_for_pin.sh --wire
CUDA_HOME=/usr/local/cuda-12.8 CMAKE_CUDA_ARCHITECTURES=120-real \
  ./scripts/build/build_llama_server.sh   # links clef.cpp when present
./scripts/phase/l6_pin_ladder_status.sh # apply check → already_applied
```

**API note:** tip Ollama / **b11351** uses `common_batch` / `batch.view` in
`collect_score_embeddings`. Tip-only follow-ups: **0136** (ANE
`sync_target_cross` shim) and **0137** (Laya encode → `llama_process`).

Operator ladder: [docs/llama-cpp-pin-ladder.md](../../docs/llama-cpp-pin-ladder.md).
Product: [docs/system-one-score.md](../../docs/system-one-score.md).
