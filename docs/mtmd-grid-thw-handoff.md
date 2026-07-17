# mtmd `grid_thw` forward handoff

**Audience:** llama.cpp / mtmd maintainers and zerollama operators tracing SGLang `video_spans.grid_thw` parity.

**Related:** [sglang-multimodal-borrowings.md §31](./sglang-multimodal-borrowings.md), [video-understanding.md §grid_thw](./video-understanding.md#optional-grid_thw-sglang-preprocessed-metadata), [llama/README.md](../llama/README.md).

---

## Why this exists

SGLang/Qwen-VL clients size `padded_input_ids` from a **client-side** patch grid (`img_grid_thw` / `video_grid_thw`). Zerollama expands video with ffmpeg and encodes frames with mtmd from **pixels** — resize and smart-crop policy can differ. When embed token count ≠ client grid, pretokenized layouts misalign with actual ViT output → context drift and degraded answers.

---

## Current state (zerollama Jul 2026)

| Layer | Behavior |
|-------|----------|
| **Preflight / usage** | `VisionTokensFromGridTHW([T,H,W], merge)` on `video_spans` — preflight, OpenAI `prompt_tokens_details` |
| **Expansion cache** | `grid_thw` stored with PNG frames in global + session LRU |
| **Runner payload** | `llm.ImageData.GridTHW` optional `[1,H,W]` per raster (`server/modality/grid_thw_raster.go`); client-explicit **and** server ffmpeg estimates |
| **llamarunner / mtmd** | `mtmd_bitmap_set_grid_hint` → dyn_size resize to `W*patch × H*patch`, skip smart_resize; log `grid_thw hint resize` |
| **ollama-engine** | Qwen3-VL / Qwen2.5-VL / glmocr / qwen3next `EncodeMultimodalWithGrid` same contract |
| **ViT cache** | PNG embed hash includes grid when set (same bytes + different grid → miss) |
| **Observability** | `vision grid hints` summary; **`vision grid hint match`** at Info when client grid aligns with embed count |

**Not shipped:** Non–M-RoPE / fixed-size / llava-uhd families ignore hints. Decoder M-RoPE positions still come from mtmd output dims (aligned when hint matches).

---

## Why hints matter (SGLang parity)

Qwen-VL / SGLang clients send pretokenized layouts sized from `img_grid_thw` / `video_grid_thw`. Zerollama can:

- Match **token counts** for preflight when hints are present.
- Compare hints vs mtmd embed counts (debug).

When ffmpeg sampling or resize policy differs from the client processor, **embed count ≠ hint** → padded inject layouts misalign with actual ViT tokens → quality / context drift.

Feeding hints into mtmd would let:

1. **Resize / patchify** to client `[H,W]` (or skip re-smart-resize when bytes already match).
2. **M-RoPE decoder positions** (`mtmd_image_tokens_get_decoder_pos`) align with pretokenized `padded_input_ids`.

---

## Upstream API sketch (proposed)

Minimal extension on existing mtmd bitmap path:

```c
// Optional per-bitmap grid hint [T,H,W]; NULL = derive from pixels (today).
MTMD_API mtmd_bitmap * mtmd_bitmap_init_from_buf_with_grid(
    const mtmd_context * ctx,
    const uint8_t * data, size_t len,
    const int32_t grid_thw[3],  // or NULL
    bool placeholder);

// Or attach after init:
MTMD_API void mtmd_bitmap_set_grid_hint(mtmd_bitmap * bmp, const int32_t grid_thw[3]);
```

**Contract:**

- `grid_thw[0]` temporal (frames in clip); per-frame bitmaps use `[1,H,W]`.
- When hint set, qwen2vl/qwen3vl clip graphs use `H,W` for patch grid instead of recomputing from decoded PNG dimensions.
- When hint absent, behavior unchanged.

**Go bind** (`llama/llama.go`): pass `img.GridTHW` from `MultimodalTokenize` when non-empty.

---

## Operator signals

```bash
# Hint applied (mtmd or ollama-engine):
rg 'grid_thw hint resize|vision grid hint match'

# E2E: VIDEO_AGENT_INFER_GRID_THW=1 with VIDEO_AGENT_INFER_PREPROC=1
RUN_E2E_VIDEO_AGENT_INFER=1 VIDEO_AGENT_INFER_GRID_THW=1 ./scripts/video/video_agent_infer_smoke.sh
```
