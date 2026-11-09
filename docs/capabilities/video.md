---
title: Video
description: Video understanding in chat and async video generation.
---

Video currently has **two separate capability families** in zerollama:

- **Video understanding** — give a model a clip in chat
- **Video generation** — create a clip asynchronously as a job

## Video understanding

**Status:** `beta`

### API surface

- OpenAI chat with `video_url` parts in `POST /v1/chat/completions`
- Native chat paths that carry video payloads through the multimodal stack

### What works today

- Native path expands video via **ffmpeg → PNG frames → image list**
- Capability checks require **vision + video** support
- Optional full-body proxy to **SGLang** when the deployment wants that path
- Session/global caches and preflight policy are part of the working surface

### Deep docs

- [video-understanding.md](../design/media/video-understanding.md)
- [multimodal-backends.md](../design/media/multimodal-backends.md)
- [upstream/sglang-multimodal-borrowings.md](../upstream/sglang-multimodal-borrowings.md)

## Video generation

**Status:** `beta` for the local async job surface, with some backends still `lab`-heavy

### API surface

- `POST /v1/videos`
- `GET /v1/videos/:id`
- `GET /v1/videos/:id/content`
- Companion media upload flow: `PUT/GET/DELETE /v1/media/...`

### What works today

- Local async video jobs through the embedded GPU-job / training queue
- Wan-style generation as the main local product path
- Keyframe / TI2V flows via `/v1/media`
- Defer/queue behavior integrated with the main inference-first policy

### Deep docs

- [wan-t2v.md](../design/media/wan-t2v.md)
- [ltx-t2v.md](../design/media/ltx-t2v.md)
- [video-c.md](../design/media/video-c.md)
- [media-uploads.md](../design/media/media-uploads.md)
- [upstream/wangp-borrowings.md](../upstream/wangp-borrowings.md)

## Current guidance

- Treat **video understanding** as the multimodal chat surface.
- Treat **video generation** as a long-running async job surface, not a synchronous completion.
- Use the product docs above for behavior; use roadmap/upstream docs only for open work or research.

## Related docs

- Product backlog: [media roadmap](../roadmaps/media.md)
- Research backlog: [RESEARCH_ROADMAP.md](../RESEARCH_ROADMAP.md)
