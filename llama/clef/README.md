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
| Live GGUF score | Lab synth: `l6_clef_live_smoke.sh`. Product: `l6_clef_product_convert.sh` + `l6_clef_decisions_e2e.sh` (PASS on tip) |
| Loader skip | `llama/compat` `translate_metadata` skips `clef.*` when `*.decision.type=clef` |

```bash
./scripts/phase/stage_clef_for_pin.sh --wire
CUDA_HOME=/usr/local/cuda-12.8 CMAKE_CUDA_ARCHITECTURES=120-real \
  ./scripts/build/build_llama_server.sh   # links clef.cpp when present
./scripts/phase/l6_pin_ladder_status.sh # apply check → already_applied
./scripts/phase/l6_clef_live_smoke.sh   # tip score_fields wire (lab port 18086)
```

**API note:** tip Ollama / **b11351** uses `common_batch` / `batch.view` in
`collect_score_embeddings`. Tip-only follow-ups: **0136** (ANE
`sync_target_cross` shim) and **0137** (Laya encode → `llama_process`).

**Product convert** (Cloudflare HF tree with `joint_head.safetensors`):

```bash
hf download Cloudflare/clef-flash --local-dir /root/models/clef-flash-hf
./scripts/phase/l6_clef_product_convert.sh
# → /root/models/clef-flash-gguf/clef-flash-ollama-q8_0.gguf (qwen35.decision.type=clef, clef.*)
```

Do **not** use `ggml-org/Clef-Flash-GGUF` with this head — that pack is native `clef` + `decision.*` tensors.

Operator ladder: [docs/llama-cpp-pin-ladder.md](../../docs/llama-cpp-pin-ladder.md).
Product: [docs/system-one-score.md](../../docs/system-one-score.md).
