# Video understanding (VLM)

This document explains **why** Ollama’s video-in-chat behavior is shaped the way it is, and how to operate it. For environment variables and manifest keys, see also [multimodal-backends.md](./multimodal-backends.md).

## Goals

1. **OpenAI compatibility** — Clients that send `video_url` parts in `POST /v1/chat/completions` should work without a separate Ollama-only contract.
2. **Work with existing vision runners** — The multimodal stack already consumes a **flat list of images** for the next completion. Reusing that path avoids a second projector protocol in v1.
3. **Predictable resource use** — Video is unbounded; **caps** (bytes, frames, images per message) exist so one request cannot exhaust disk, RAM, or context by default.
4. **Optional SGLang** — Some deployments already run **SGLang** for VLMs. Proxying the full chat body preserves parity with upstream behavior **without** making SGLang mandatory for local use.

## Why one merged user message per OpenAI turn

OpenAI represents a single logical user turn as a `content` **array** (text, `image_url`, `video_url`, …). Splitting that into multiple `api.Message` rows duplicated the same role and made **ordering** between still images and video frames ambiguous.

**We merge** into one `api.Message` with:

- `content`: text segments joined with newlines (same as typical multi-block text).
- `images`: still images and `input_audio` payloads in **array order** (unchanged semantics for audio-in-images).
- `videos`: raw container bytes per `video_url`, in order, **before** expansion.

**Why** separate `videos` before expansion: the runner does not consume raw video; **ffmpeg** needs a blob. Staging bytes on `Videos` and expanding in **one place** (`ExpandVideosInChatRequest`) keeps OpenAI parsing simple and centralizes limits/errors.

## Why ffmpeg → PNG frames → `images`

The llama multimodal path expects **raster image bytes** for vision towers. Decoding arbitrary containers and sampling time-uniform frames is what **ffmpeg** is for; outputting **PNG** gives a consistent format for the rest of the pipeline.

**Trade-off:** Many frames look like many independent images to the template. That is **frame-list semantics** (documented in renderer notes). Models that expect a single “video” tensor would need deeper integration later.

## Why `FromChatRequestWithContext` and fetch timeouts

Remote `video_url` implies a **server-side HTTP GET**. That must:

- Respect **client disconnect** (`context` from the Gin request), so a hung download does not run forever after the user closes the tab.
- Bound **wall time** (`OLLAMA_VIDEO_FETCH_TIMEOUT`) so a pathological upstream cannot tie up a worker indefinitely.
- Use the **default HTTP transport** (cloned) so TLS, HTTP/2, and proxy env behave like the rest of the process.

**Why** `ResponseHeaderTimeout` on the transport: distinguish “upstream never answers” from “long streaming body,” without cutting off large but healthy downloads before headers arrive.

## Why HTTPS by default for remote URLs

Plain **HTTP** hides no transport security. Defaulting to **HTTPS** reduces accidental cleartext use on the public internet. **Local or lab** setups can set `OLLAMA_VIDEO_ALLOW_INSECURE_HTTP=1` when they intentionally use `http://`.

## Why SSRF checks are “best effort”

Before `GET`, we **resolve** the hostname and **reject** loopback, private, and link-local addresses. That blocks the common “point Ollama at `http://169.254.169.254/`” class of abuse.

**Why** this is not complete: the actual TCP connection is not **cryptographically pinned** to those resolved IPs; **DNS rebinding** can theoretically serve different addresses over time. High-assurance deployments should use **allowlists**, private networks, or **data URIs** instead of arbitrary URLs.

## Why optional SGLang proxy

When `modality_backends.video_understanding` is **`sglang`** and `OLLAMA_SGLANG_URL` is set, requests that include **`video_url`** are proxied as **full JSON** to `{base}/v1/chat/completions`.

**Why** full body: partial rewriting (only video parts) would duplicate SGLang’s own OpenAI parsing and drift over time.

**Why** only when `video_url` is present: avoids sending every chat through SGLang when the model is hybrid.

**Operational note:** The **`model`** field is forwarded unchanged. SGLang must recognize that id, or you align names (Ollama tag vs SGLang model id) in your deployment.

## Capability checks (`vision` + `video`)

**Why** both: `vision` means the model can consume image tensors; `video` signals that the manifest/stack is intended for **video-understanding** flows (including expanded frame count). A model without vision cannot use video frames; failing early returns **400** with a clear error instead of obscure runtime failures.

Checks run when a message carries raw **`videos`** or pre-expanded **`video_spans`** (SGLang-style clients that skip ffmpeg).

## Preflight and multi-turn history

Vision budget preflight counts **raw `videos` on any message** (about to expand) and **pre-expanded `video_spans` / audio only on the latest user message**. **Why:** agents often echo full chat history; counting old expanded frames would false-reject valid follow-up turns. Text-only history is unaffected.

## Session cache keys (OpenAI + native)

Session video expansion cache and L3 prefix reuse need a stable thread id:

| API | How to pass |
|-----|-------------|
| `/api/chat` | `options.prompt_cache_key` or `options.eliza.conversationId` |
| `/v1/chat/completions` | top-level `prompt_cache_key` or `options.prompt_cache_key` |
| Responses API | `prompt_cache_key` field (existing) |

Without a key, global expansion LRU still helps; per-thread session cache does not.

## Why policy is resolved once on the server

Sampling limits are merged from **env** (fleet-wide defaults) and the model **`config.json`** (per-artifact tuning). Resolution happens in the HTTP handler so **`server/modality`** does not import `server.Model`: that keeps import cycles out and makes policy easy to test. The same **VideoSamplingPolicy** value is used for preflight and expansion so behavior cannot drift between checks.

## Native sampling policy (fps vs stride)

Server defaults come from environment variables; **per-model** overrides live in `config.json` under **`video_sampling`** (see [multimodal-backends.md](./multimodal-backends.md)).

**Why two modes:** **fps** matches “sample uniformly in time” (good for clips where wall-clock matters). **stride** matches “every Nth decoded frame” (good when you care about frame index density, not wall-clock). Both are intentional **lossy compression** of video into a bounded frame list.

| Mode | Behavior |
|------|----------|
| **`fps`** (default) | Time-uniform sampling via ffmpeg’s `fps` filter (`OLLAMA_VIDEO_SAMPLE_FPS` or manifest `fps`). |
| **`stride`** | Emit every Nth **decoded** frame (`OLLAMA_VIDEO_STRIDE` or manifest `stride`); uses ffmpeg `select` + `setpts`. |

**Caps:** `max_frames` (env `OLLAMA_VIDEO_MAX_FRAMES` or manifest) still limits how many PNGs are kept after filtering.

## Failure modes (native path)

| Symptom | Typical cause |
|---------|----------------|
| `ffmpeg: ...` stderr in the error | Unsupported/corrupt container, missing codecs, or invalid filter; ensure the blob is a format ffmpeg can demux. |
| `ffmpeg produced no frames` | Empty stream, no decodable video track, or filters removed every frame. |
| `video exceeds max size` | Raise `OLLAMA_VIDEO_MAX_BYTES` or shrink the input. |
| `too many images after video expansion` | Lower `OLLAMA_VIDEO_MAX_FRAMES` / `OLLAMA_VIDEO_MAX_IMAGES_PER_MESSAGE` or use fewer clips. |
| `estimated vision tokens ... exceed num_ctx` | **Preflight** upper bound: raw **`videos`** on any message (stills + `videos × max_frames × tokens_per_image`); pre-expanded **`video_spans`** / **audio** on the **latest user** message only (default **768** per frame; manifest **`tokens_per_image`**). Increase **`num_ctx`**, reduce frames, or remove clips. |

**Why preflight scopes history:** Full prompt budgeting would duplicate truncate/shift. Raw video on this request always counts. Pre-expanded spans on old user turns are skipped — agents echo history and would otherwise false-reject innocent follow-ups.

**Why preflight ignores older turns’ still images:** Same as above; historical stills may be truncated away before render.

**Why preflight does not count text tokens:** The goal is a **cheap** fail before ffmpeg and disk; tokenizing the whole history here would add latency. Text can still push you over `num_ctx`; use truncate/shift or raise `num_ctx` as usual.

## mllama / single-image models

Some **mllama** vision stacks accept **only one image per message**. **Preflight** rejects raw video that would expand to multiple frames before ffmpeg runs. Historical user turns with echoed multi-frame expansions are skipped when only the latest turn is text-only; if history still carries multiple images per message, prompt construction fails as before. Prefer **one short clip**, **`max_frames=1`**, or a multi-image VLM.

## Why successful sampling is logged at Info

After ffmpeg runs, the server logs effective **mode**, **fps/stride**, **max_frames**, **frame_count**, and whether the manifest overrode defaults. **Why Info (not only Debug):** operators troubleshooting production need to see what actually ran without enabling verbose logging for the whole server.

Cache hits log separately:

| Event | Why it matters |
|-------|----------------|
| `video url fetch cache hit` | Same HTTPS URL — container bytes reused (hostname only in logs) |
| `video sample session cache hit` | Same `prompt_cache_key` + clip — ffmpeg skipped for this agent thread |
| `video sample global cache hit` | Same policy + video digest — ffmpeg skipped (any client) |
| `video sample` | ffmpeg actually ran |

**Why three expansion layers:** URL cache saves network; global cache shares wins across users; session cache keeps clips warm for one agent when the global LRU (32 entries) evicts under fleet load. Session cache requires a stable thread key — `options.prompt_cache_key`, eliza `conversationId` / `promptCacheKey`, or OpenAI `/v1/chat/completions` **`prompt_cache_key`** — same semantics as [L3 prefix cache](gpu-profiles-l3.md). See [sglang-multimodal-borrowings.md](./sglang-multimodal-borrowings.md).

## OpenAI usage: modality breakdown and `cached_tokens`

Streaming and final chat/generate responses can include:

```json
"usage": {
  "prompt_tokens": 1280,
  "completion_tokens": 42,
  "prompt_tokens_details": {
    "image_tokens": 768,
    "video_tokens": 2304,
    "cached_tokens": 120
  }
}
```

**Why `image_tokens` / `video_tokens`:** OpenAI and SGLang expose per-modality prompt estimates for billing and debugging. Zerollama computes an **upper-bound heuristic** after video expansion (`tokens_per_image` from manifest, default 768 per still frame / sampled frame / audio clip). These are not exact vision-tower counts.

**Why `cached_tokens`:** L3 `cache_prompt` and llama-server slot reuse report `timings.cache_n`. Without surfacing it, prefix KV hits are invisible in the API and access logs (`cached_prompt_tokens` on `inference response out`).

**Live gate:** `RUN_E2E_VIDEO_AGENT_INFER=1 ./scripts/video_agent_infer_smoke.sh` — two-turn VLM inference with the same `prompt_cache_key`; strict pass requires turn-2 `cached_prompt_tokens` ≥ 1 on `/api/chat` (L3 KV on llama-server subprocess, **or ollama-engine input-cache hits** on Mac Metal). OpenAI `/v1/chat/completions` in the same script is advisory (single-turn, different prefix). Use `VIDEO_AGENT_INFER_SOFT=1` on MLX-only or when both `llama_cache.enabled` and runner KV cache are off. Optional legs (require `VIDEO_AGENT_GO_LOG`): **`VIDEO_AGENT_INFER_PREPROC=1`** (pre-expanded padded infer; `preprocessed layout session cache hit`); **`VIDEO_AGENT_INFER_PREFIX_MM_WARN=1`** (prefix-mm hint without session key); **`VIDEO_AGENT_INFER_VIT_SESSION=1`** (strict `vision embed session cache hit` on turn 2); **`VIDEO_AGENT_INFER_GRID_THW=1`** (requires preproc; `grid_thw hint resize` or `vision grid hint match`). Grep logs for `vision grid hints`, `precomputed_embedding runner inject`, `processor_output runner inject`. Report: `./scripts/video_agent_infer_gate_report.sh /tmp/video-agent-infer-smoke.json`.

**Fleet logs:** the same `inference response out` line includes `image_tokens`, `video_tokens`, and `audio_tokens` when post-expand heuristics are non-zero — mirrors `prompt_tokens_details` without parsing JSON bodies.

## Preprocessed messages (`video_spans` without `videos`)

If a client sends **`images` + `video_spans`** and no raw **`videos`**, expansion is skipped — the message is treated as already processed (SGLang-style fast path). **Why:** double ffmpeg on the same frames wastes CPU and disk.

**Validation:** the sum of `video_spans[].frame_count` must not exceed `len(images)`; otherwise the handler returns **400** before render.

## `video_spans` on `api.Message`

After expansion, **`video_spans`** lists **`frame_count` per original `videos[]` entry** (in order). **Still images** come first in **`images`**, then frames for each video in order. Renderers that need to distinguish “N frames from one clip” from unrelated images can use this metadata; token layout may still match a flat image list (see Qwen3-VL renderer notes).

### Optional `grid_thw` (SGLang preprocessed metadata)

Pre-expanded clients may attach **`grid_thw`: `[T, H, W]`** on each `video_spans[]` entry (Qwen/SGLang patch grid). **Why:** the flat `768 × frame_count` heuristic over-estimates or under-estimates vs real vision towers; `grid_thw` lets preflight and `usage.prompt_tokens_details` use `T×H×W / spatial_merge²` (default merge 2) without running the vision processor in Go.

**Validation:** `len(grid_thw) == 3`, all positive, and `grid_thw[0]` must equal `frame_count` when both are set. Invalid grids return **400** before render.

### Optional `padded_input_ids` (SGLang pretokenized layout — partial runner consume)

Pre-expanded clients may send **`padded_input_ids`** on the latest user message alongside **`images` + `video_spans`**, or on a single-clip message with raw **`videos[]`** (turn 1). **Why:** SGLang clients already have the full pretokenized prompt slice; zerollama uses `len(padded_input_ids)` for **preflight** and **usage** on that turn instead of frame×768 heuristics. **Layout cache:** on single-clip messages with `options.prompt_cache_key`, `padded_input_ids` is stored in the **session** expansion LRU (not global — layout is client-specific). Restore paths: (1) raw `videos[]` resend on turn 2 (`video layout session cache hit`); (2) pre-expanded `images` + `video_spans` on turn 2 (`preprocessed layout session cache hit`, fingerprinted by frame bytes + span metadata). **Runner (Qwen3-VL):** routes splice **each** user turn's `padded_input_ids` into matching `<|im_start|>user` blocks when template alignment holds (multi-turn agent). Tool results render as pseudo `<|im_start|>user\n<tool_response>…` blocks — they are **excluded** from splice span counting so tool-calling agents do not break alignment. Renderer skips duplicate vision placeholders or `[img-N]` tags on the latest padded user; prior padded users keep placeholders when `role=tool` exists (safe fallback). llamarunner injects full mtmd chunks per `<|vision_start|>…<|vision_end|>` block; llama-server maps vision blocks to media markers (`layout_consume=qwen3vl_hf_runner_inject`). On splice failure: `deferred_multimodal_history` + warn log `padded_input_ids splice failed`. See [sglang-multimodal-borrowings.md](./sglang-multimodal-borrowings.md) §25.

**Validation:** non-negative ids, max length 1M. Invalid values return **400** before render.

### Optional `precomputed_embedding` (SGLang ViT output rows)

Clients may send **`precomputed_embedding`** (post-projector feature rows) with **`padded_input_ids`** instead of PNG **`images[]`**. **Why:** when the client or a GPU worker already ran the vision tower, the server should splice rows at pretokenized vision slots — not decode PNG and re-encode. **Rules:** exactly one precomputed item per message; cannot mix with raw bytes or `processor_output`; requires `padded_input_ids` on the same message. **ollama-engine** implements all native Go VLMs (Qwen/glmocr need `grid_thw` patch grid; Gemma/mllama/deepseekocr rows only; llama4 optional multi-tile grid; lfm2 single-tile). **ggml llamarunner** accepts embed rows on padded inject paths. **llama-server** rejects (400). Grep `precomputed_embedding runner inject`.

### Optional `processor_output` (SGLang HF processor tensors)

Clients may send **`processor_output`** with **`pixel_values`** and **`image_grid_thw`** plus **`padded_input_ids`**. **Why:** SGLang runs the HF processor locally; zerollama should run only vision tower + projector + LLM — skipping PNG decode and processor CPU work. **Rules:** same exclusivity as precomputed; one item per message. **ollama-engine:** Qwen/glmocr patch grids; Gemma3/4/mistral3/llama4/lfm2 pixel canvases (`[1,H,W]`; llama4/lfm2 single-tile only). **deepseekocr** deferred (SAM pipeline). **llamarunner / llama-server** reject. Grep `processor_output runner inject`.

### `enable_prefix_mm_cache` (session ViT pin)

Set **`prompt_cache_key`** on every agent turn. Session ViT overlay defaults **ON** when the key is set; set **`enable_prefix_mm_cache: false`** to disable session pin and rely on global LRU only. **Why:** ffmpeg and expansion caches can hit while the runner's global ViT LRU (4–64 slots) still evicts embeds between agent turns — SGLang keeps encoder outputs hot per conversation; zerollama's session overlay uses the same key as L3 and session expansion. Without `prompt_cache_key`, overlay stays off (optional warn log when the flag is set alone).

**Logs:** `vision embed session cache hit`, `vision embed global cache hit`, `precomputed_embedding session cache hit` (llamarunner).

### Native ffmpeg `grid_thw`

Native ffmpeg expansion **populates `grid_thw`** when sampled PNG frames decode: first-frame dimensions → Qwen3-VL-style smart resize (patch 14, merge 2 by default) → `[frame_count, H_patch, W_patch]`. Override via manifest `vision_patch_size` / `vision_spatial_merge_size`. **Layout cache:** `grid_thw` is stored in global + session expansion LRU with frames — agent turn 2 skips PNG re-decode for grid **estimates**.

**Runner forward (Jun 2026):** only **client-origin** `grid_thw` on pre-expanded `video_spans` is forwarded to mtmd (`GridTHWExplicit`). Server ffmpeg estimates are **preflight/usage only** — Go smart_resize can differ from mtmd. On M-RoPE models (Qwen-VL), explicit `[1,H,W]` per frame triggers `mtmd_bitmap_set_grid_hint` resize (`grid_thw hint resize` in logs). Grep `vision grid hint match` when client grid aligns with mtmd output.

**precomputed / processor_output:** client `grid_thw` on those payloads always forwards to ollama-engine (no VideoSpan gate).

**Why:** native path token estimates should match preprocessed clients without requiring SGLang upstream. Other vision families may differ; ffmpeg estimates are heuristic only.

## Optional external decode hook (Phase E)

For a **narrow** subprocess replacing in-process ffmpeg only, set **`modality.ExternalVideoDecodeHook`** (Go API) in a fork or plugin build—see [ROADMAP.md](./ROADMAP.md) Phase E. The default remains ffmpeg inside the server process.

## Parity matrix

See **[video-parity.md](./video-parity.md)** for reference workloads and a native vs optional-SGLang comparison table.

## Related documentation

- [multimodal-backends.md](./multimodal-backends.md) — env vars and `modality_backends` keys.
- [sglang-multimodal-borrowings.md](./sglang-multimodal-borrowings.md) — what was ported from SGLang and why.
- [ROADMAP.md](./ROADMAP.md) — **Option 2:** in-tree milestones to narrow parity with external stacks **without** SGLang; [Wan T2V](./wan-t2v.md) (generation) vs this doc (understanding); hardening.
