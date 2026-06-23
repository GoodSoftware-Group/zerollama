# Video parity matrix (native VLM vs optional SGLang)

This document defines **reference workloads** and a **parity matrix** for [Option 2 in ROADMAP.md](./ROADMAP.md): closing the gap on the **native** path (ffmpeg + vision runner) without requiring SGLang.

**Why a matrix:** “parity with SGLang” is not one boolean—different rows (sampling, context, templates) progress independently. Named rows and columns make gaps explicit so PRs can claim **which** behavior changed, and **why** a row is left empty (not pursued).

**Why reference workloads:** Reproducible clips and model tags turn “it feels better” into **testable** acceptance criteria for native sampling and limits.

Empty cells mean **not pursued** for that column. “Target” is deployment-specific—fill in for your models.

## Reference workloads

Use at least one short clip and one vision model family you care about:

| ID | Model family (example) | Input | Success criteria |
|----|-------------------------|-------|------------------|
| W1 | Qwen3-VL (or similar VLM) | MP4 or WebM, a few seconds, H.264/VP9 | Chat completes; frames sample without ffmpeg error; output is coherent on a simple visual question |
| W2 | Same as W1 | Slightly longer clip or higher resolution | Stays within **max bytes**, **max frames**, and **num_ctx** after expansion (no silent OOM) |

Document the exact model tag and clip source in your issue or PR when changing sampling defaults.

## Parity matrix

**Rows** are behaviors operators care about. **Columns**:

| Column | Meaning |
|--------|---------|
| **Native (today)** | Current in-tree behavior before your PR |
| **Native (after M1)** | Per-manifest `video_sampling`, env overrides, structured logs, tests |
| **Optional SGLang** | `modality_backends.video_understanding=sglang` + `OLLAMA_SGLANG_URL`; full-body proxy when `video_url` is present |
| **Target** | What you need for your deployment (fill in) |

| Behavior | Native (today) | Native (after M1) | Optional SGLang | Target |
|----------|----------------|-------------------|-----------------|--------|
| Time-uniform sampling (fps) | `OLLAMA_VIDEO_SAMPLE_FPS` + max frames | + per-model `video_sampling.mode=fps` | Upstream stack | |
| Stride / nth-frame sampling | env + manifest | `mode=stride` + stride | Upstream stack | |
| Max frames cap | `OLLAMA_VIDEO_MAX_FRAMES` | + manifest `max_frames` | Upstream stack | |
| Byte / message limits | `OLLAMA_VIDEO_*` env | unchanged | proxy body limits | |
| Failure modes (corrupt video, no frames) | ffmpeg error → 400 | documented + logs | upstream | |
| Context budget before decode | preflight vs `num_ctx`; raw `videos` any message; pre-expanded spans on **latest user** only | + `tokens_per_image` manifest | upstream scheduler | |
| Capability gate | raw `videos` **or** `video_spans` | vision + video caps | upstream | |
| mllama single-image constraint | preflight before ffmpeg + post-expand prompt check | documented; same or downsample (policy) | n/a | |
| Template: flat images vs video spans | flat `images` + `video_spans` metadata | Gemma4 HF placeholders (`!RenderImgTags`) | upstream | |
| Repeat `video_url` fetch | pooled HTTP + URL body LRU (32 / 30 m) | — | upstream CDN cache | |
| Repeat clip ffmpeg skip | global expand LRU + session LRU w/ `prompt_cache_key` | — | upstream preprocessor cache | |
| Preprocessed frames (no raw video) | skip ffmpeg when `video_spans` set | span validation | `processor_output` path | |
| OpenAI usage modality breakdown | `image_tokens`, `video_tokens`, `audio_tokens` (heuristic) | — | upstream | |
| OpenAI session cache key | `/v1/chat/completions` `prompt_cache_key` + `options`; `/api/chat` `options` | — | Responses API field | |
| Prefix KV visibility | `cached_tokens` in `prompt_tokens_details` | access log `cached_prompt_tokens` | upstream | |
| Modality token visibility | `image_tokens`, `video_tokens`, `audio_tokens` (heuristic) | access log same fields on `inference response out` | upstream | |
| Full `grid_thw` / pretokenized layout cache | `grid_thw` global+session LRU; `padded_input_ids` session LRU; Qwen3-VL **multi-turn** runner inject + tool-span splice | full processor consume | upstream |

## How to use this in PRs

- For native-path changes, update the **Native (after M1)** column or add a footnote when behavior is intentional.
- Do not claim full parity with **every** SGLang model; the **Optional SGLang** column is for comparison only.

See also [video-understanding.md](./video-understanding.md), [sglang-multimodal-borrowings.md](./sglang-multimodal-borrowings.md), and [multimodal-backends.md](./multimodal-backends.md).
