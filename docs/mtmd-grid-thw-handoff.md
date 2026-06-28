# mtmd `grid_thw` forward handoff

**Audience:** llama.cpp / mtmd maintainers and zerollama operators tracing SGLang `video_spans.grid_thw` parity.

**Related:** [sglang-multimodal-borrowings.md §31](./sglang-multimodal-borrowings.md), [video-understanding.md §grid_thw](./video-understanding.md#optional-grid_thw-sglang-preprocessed-metadata), [llama/README.md](../llama/README.md).

---

## Why this exists

SGLang/Qwen-VL clients size `padded_input_ids` from a **client-side** patch grid (`img_grid_thw` / `video_grid_thw`). Zerollama expands video with ffmpeg and encodes frames with mtmd from **pixels** — resize and smart-crop policy can differ. When embed token count ≠ client grid, pretokenized layouts misalign with actual ViT output → context drift and degraded answers.

**Why not block hints until upstream lands:** operators need preflight/usage parity and debug visibility (`vision grid hints`) today; blocking the Go wire would force a second refactor when mtmd adds `mtmd_bitmap_set_grid_hint`. The seam accepts `gridTHW` now, logs at Debug when present, and leaves pixel-derived encode unchanged.

---

## Current state (zerollama Jun 2026)

| Layer | Behavior |
|-------|----------|
| **Preflight / usage** | `VisionTokensFromGridTHW([T,H,W], merge)` on `video_spans` — preflight, OpenAI `prompt_tokens_details` |
| **Expansion cache** | `grid_thw` stored with PNG frames in global + session LRU |
| **Runner payload** | `llm.ImageData.GridTHW` optional `[1,H,W]` per raster (`server/modality/grid_thw_raster.go`) |
| **llamarunner seam** | `MtmdContext.MultimodalTokenize(..., gridTHW)` forwards hint via `mtmd_bitmap_set_grid_hint` |
| **mtmd forward (Jun 2026)** | M-RoPE models: hint resize to `W*patch x H*patch`, skip `smart_resize`; log `grid_thw hint resize` |
| **Observability** | `vision grid hints` summary; **`vision grid hint match`** at Info when client grid aligns with mtmd output |

**Client vs server grid:** Only **client-origin** `grid_thw` on pre-expanded `video_spans` is forwarded to mtmd (`VideoSpan.GridTHWExplicit`). Server ffmpeg estimates populate `grid_thw` on spans for **preflight/usage** but do not override ViT resize (Go smart_resize ≠ mtmd smart_resize).

**Not shipped:** M-RoPE **decoder positions** still derived from mtmd output dimensions (aligned when hint matches). Non–M-RoPE families ignore hints. llava-uhd tiling unchanged.

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

## Zerollama wiring checklist (post-upstream)

1. Bump `LLAMA_CPP_VERSION` + regen Metal embed if needed.
2. ~~Replace debug log in `llama.MtmdContext.MultimodalTokenize(..., gridTHW)` with `mtmd_bitmap_set_grid_hint`~~ **Done (Jun 2026)** — only **client explicit** grids forwarded via `GridTHWPerRaster` + `VideoSpan.GridTHWExplicit`.
3. Remove or downgrade `vision grid hint mismatch` logs once hints are authoritative.
4. E2E: `RUN_E2E_VIDEO_AGENT_INFER=1` with pre-expanded `video_spans` + `grid_thw` on Qwen3-VL; optional `VIDEO_AGENT_INFER_PREPROC=1` for padded infer leg.

---

## Operator signals today

```bash
# Debug: compare client grid vs mtmd embed count per frame
OLLAMA_DEBUG=1 ollama serve 2>&1 | rg 'vision grid hint'

# Preflight already uses grid when present on video_spans
curl .../api/chat -d '{"messages":[{"role":"user","video_spans":[{"frame_count":4,"grid_thw":[4,24,32]}],...}]}'
```

Mismatch logs are **expected** until upstream accepts hints — mtmd still pixel-derives layout.
