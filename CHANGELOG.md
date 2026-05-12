# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Zerollama → Eliza Cloud (default remote inference):** default upstream `https://www.elizacloud.ai`, `ELIZACLOUD_API_KEY` sent as `X-API-Key` on `/api/v1/...`; **Ed25519 request signing only** when `OLLAMA_CLOUD_BASE_URL` targets `ollama.com` (legacy cloud). Client paths `/v1/*` are rewritten to Eliza `/api/v1/*`; `/api/embed` and `/api/embeddings` map to `/api/v1/embeddings`. **Why:** OpenAI/Anthropic-compatible APIs and API-key auth match how agents integrate; legacy signing stays opt-in for ollama.com users.
- **Cloud model catalog merge:** `GET /api/v1/models` merged into local tag lists when cloud is enabled, with **singleflight** on fetch, **Cache-Control**–aware TTL (clamped), and dedupe by model name. **Why:** one combined list for operators; avoids stampedes and duplicate rows.

- **Native video sampling policy:** env `OLLAMA_VIDEO_SAMPLE_MODE` / `OLLAMA_VIDEO_STRIDE`, optional manifest `video_sampling` and `tokens_per_image`, centralized ffmpeg filter builder, structured **Info** logs after sampling, **`video_spans`** on `api.Message`, context **preflight** against `num_ctx` (messages with video only), and **[video-parity.md](docs/video-parity.md)** (Option 2 matrix).
- **Video understanding (VLM)** for OpenAI-compatible chat: `content` parts with `type: "video_url"` are merged into a single user message, decoded (data URI or remote HTTPS by default), sampled to frames via **ffmpeg**, and fed through the existing vision path as additional images (`docs/video-understanding.md`).
- **`api.Message.videos`** for raw video bytes on `POST /api/chat`; expansion runs before prompt rendering.
- **Manifest / capabilities:** `modality_backends.video_understanding` values `native` (default) or `sglang`; **`video`** capability alongside vision where applicable.
- **Optional SGLang proxy:** when `video_understanding=sglang` and `OLLAMA_SGLANG_URL` is set, `POST /v1/chat/completions` bodies that include `video_url` can be forwarded in full to SGLang’s `/v1/chat/completions`.
- **Environment variables** for limits and behavior: `OLLAMA_FFMPEG`, `OLLAMA_SGLANG_URL`, `OLLAMA_VIDEO_*` (see `docs/multimodal-backends.md`).
- **`FromChatRequestWithContext`** so remote `video_url` fetches respect request cancellation; `FromChatRequest` remains for callers without a context.

### Security

- Remote `video_url` fetches use **HTTPS by default**; `http://` requires `OLLAMA_VIDEO_ALLOW_INSECURE_HTTP=1`.
- **SSRF mitigation:** DNS resolution before GET with rejection of loopback/private/link-local targets (see `docs/video-understanding.md` for limitations).

### Changed

- **Eliza outbound auth:** `X-API-Key` is applied to all proxied paths toward non-`ollama.com` upstreams (not only `/api/v1/...`); missing key logs **once** per process on first such request. **Path rewrite:** only `/v1` and `/v1/...` are mapped to `/api/v1/...` (avoids mangling paths like `/v1chat`). **Signing:** Ed25519 uses `isOllamaComUpstream()` instead of a redundant `signingHost` return value from `OLLAMA_CLOUD_BASE_URL` resolution.
- OpenAI multimodal `content` arrays are converted to **one** internal `api.Message` per assistant/user turn (text + images + videos) instead of multiple messages per part, preserving array order for vision inputs.
- **Native video:** invalid manifest `video_sampling.mode` logs a warning and falls back to **fps**; **`ExternalVideoDecodeHook`** runs only after empty/size checks (same as ffmpeg path).

### Documentation

- **[eliza-cloud.md](docs/eliza-cloud.md)** — **why** Eliza is the default upstream, **why** `X-API-Key` vs Ed25519 signing, path rewrites, catalog merge/cache, raw upstream JSON on some routes, account stubs off ollama.com.
- **[ROADMAP.md](docs/ROADMAP.md)** — **why** the roadmap file exists; **Option 2** video phases; **[Zerollama remote cloud (Eliza)](docs/ROADMAP.md#zerollama-remote-cloud-eliza)** follow-ups and non-goals.
- **[video-understanding.md](docs/video-understanding.md)** — **why** merged OpenAI messages, ffmpeg→PNG, **why** preflight scopes to messages with video, **why** `video_spans`, logging at Info.
- **[multimodal-backends.md](docs/multimodal-backends.md)** — **why** env + manifest both apply to sampling.
- **[video-parity.md](docs/video-parity.md)** — **why** a parity matrix and reference workloads.
- **[README.md](README.md)** — in-repo doc links with short rationale (Eliza Cloud + video).
- Code comments in **`server/cloud_proxy.go`** / **`server/eliza_catalog.go`** (remote proxy defaults, path rewrite, singleflight) and **`server/modality`** (video policy, preflight, ffmpeg, expansion) plus **`types/model` / `api`** types where relevant — **why** decisions, not only **what**.
