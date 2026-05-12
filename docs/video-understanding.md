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
| `estimated vision tokens ... exceed num_ctx` | **Preflight** upper bound **only for messages that include `videos`**: still images on those turns plus `videos × max_frames × tokens_per_image` (default **768** per frame; override with manifest **`tokens_per_image`**). Earlier turns with images but no video are not counted here. Increase **`num_ctx`**, reduce frames, or remove images. |

**Why preflight ignores older turns’ images:** Full prompt budgeting would require duplicating truncate/shift logic. Counting only turns that contain **video** catches “this request will obviously blow the window” after expansion without rejecting long chats that only added images in the past (those turns may be truncated away).

**Why preflight does not count text tokens:** The goal is a **cheap** fail before ffmpeg and disk; tokenizing the whole history here would add latency. Text can still push you over `num_ctx`; use truncate/shift or raise `num_ctx` as usual.

## mllama / single-image models

Some **mllama** vision stacks accept **only one image per message**. After expansion, **multiple sampled frames** count as multiple images, so chat may fail with an error from prompt construction. Prefer **one short clip**, **lower max frames**, or a model that supports multi-image / video layouts. Automatic downsampling is not implied—treat this as a **policy** decision for your deployment.

## Why successful sampling is logged at Info

After ffmpeg runs, the server logs effective **mode**, **fps/stride**, **max_frames**, **frame_count**, and whether the manifest overrode defaults. **Why Info (not only Debug):** operators troubleshooting production need to see what actually ran without enabling verbose logging for the whole server.

## `video_spans` on `api.Message`

After expansion, **`video_spans`** lists **`frame_count` per original `videos[]` entry** (in order). **Still images** come first in **`images`**, then frames for each video in order. Renderers that need to distinguish “N frames from one clip” from unrelated images can use this metadata; token layout may still match a flat image list (see Qwen3-VL renderer notes).

## Optional external decode hook (Phase E)

For a **narrow** subprocess replacing in-process ffmpeg only, set **`modality.ExternalVideoDecodeHook`** (Go API) in a fork or plugin build—see [ROADMAP.md](./ROADMAP.md) Phase E. The default remains ffmpeg inside the server process.

## Parity matrix

See **[video-parity.md](./video-parity.md)** for reference workloads and a native vs optional-SGLang comparison table.

## Related documentation

- [multimodal-backends.md](./multimodal-backends.md) — env vars and `modality_backends` keys.
- [ROADMAP.md](./ROADMAP.md) — **Option 2:** in-tree milestones to narrow parity with external stacks **without** SGLang; video generation; hardening.
