---
name: video-understanding-chat
description: "Send video clips into a zerollama chat request for vision-language understanding (video_url/videos in /v1/chat/completions), distinct from generating video."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, video, vlm, vision, multimodal, ffmpeg]
    category: mlops
    related_skills: [zerollama-integration, generate-video]
---

# Video Understanding (Chat) Skill

Send a video clip into a chat request so a vision-language model on
[zerollama](https://github.com/GoodSoftware-Group/zerollama) can describe,
answer questions about, or reason over its contents. This is video **input**
(understanding) — the opposite of `generate-video` (Wan text-to-video
**output**). Zerollama decodes the video into a bounded list of frames
(ffmpeg → PNG) and feeds them through the existing vision pipeline; there is
no separate "video tensor" model interface.

## When to Use

- The user wants a model to summarize, describe, or answer questions about
  a video clip
- Choosing frame-sampling settings (fps vs. stride) for a clip
- Debugging `ffmpeg` errors, "too many images after video expansion", or
  context-overflow errors from a video-containing chat request

## Prerequisites

- zerollama server running with `ffmpeg` available on the host
- A **vision-capable** model with the **`video`** capability manifest flag
  (not just `vision`) — a vision-only model without the video flag returns
  a clear `400` instead of an obscure runtime failure

## API Contract

Send video the same way OpenAI's `video_url` content part works, in
`POST /v1/chat/completions`:

```json
{
  "model": "some-vlm",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "What is happening in this clip?"},
      {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}}
    ]
  }]
}
```

Server-side, the video is fetched (HTTPS only by default — see Pitfalls),
sampled into a bounded frame list, and merged into the same `images` array
the vision pipeline already consumes. `POST /api/chat` accepts the native
equivalent (`videos` field).

## Sampling controls

| Mode | Behavior | Config |
|---|---|---|
| `fps` (default) | Time-uniform sampling | env `OLLAMA_VIDEO_SAMPLE_FPS` or manifest `video_sampling.fps` |
| `stride` | Every Nth decoded frame | env `OLLAMA_VIDEO_STRIDE` or manifest `video_sampling.stride` |

Caps: `OLLAMA_VIDEO_MAX_BYTES` (input size), `OLLAMA_VIDEO_MAX_FRAMES`
(frames kept after sampling), `OLLAMA_VIDEO_MAX_IMAGES_PER_MESSAGE`.

## Session caching for multi-turn agents

Pass a stable thread id so repeated turns don't re-run ffmpeg on the same
clip:

| API | Field |
|---|---|
| `/api/chat` | `options.prompt_cache_key` or `options.eliza.conversationId` |
| `/v1/chat/completions` | top-level `prompt_cache_key` |

Without a key, only the global (cross-user) expansion cache helps.

## Pitfalls

- **Model needs `video`, not just `vision`** — some manifests only flag
  `vision`; check `/api/show` capabilities before assuming a VLM handles
  clips.
- **HTTPS only by default** — plain `http://` URLs are rejected unless
  `OLLAMA_VIDEO_ALLOW_INSECURE_HTTP=1` (lab/local only).
- **SSRF guard blocks loopback/private/link-local hosts** — pointing
  `video_url` at internal infra will be rejected; this is best-effort (not
  DNS-rebinding-proof), don't rely on it as a full security boundary.
- **`estimated vision tokens ... exceed num_ctx`** — raw `videos` on *any*
  message count toward preflight (stills + `videos × max_frames ×
  tokens_per_image`, default 768/frame); increase `num_ctx`, cut
  `max_frames`, or drop clips from echoed history.
- **mllama / single-image models** — some vision stacks accept only one
  image per turn; multi-frame video is rejected up front. Use `max_frames=1`
  or a multi-image-capable VLM.
- **Optional SGLang proxy** — when `modality_backends.video_understanding:
  sglang` is set, the **whole** chat body is forwarded to SGLang's
  `/v1/chat/completions`; the `model` field must match an id SGLang
  recognizes.
- **Don't confuse with `generate-video`** — this skill is about a model
  *watching* a clip you provide; generating a new clip from text is a
  completely different async job API (`/v1/videos`).

## Related

- `zerollama-integration` — generic API contract, sizing, pitfalls
- `generate-video` — the inverse: generating a video clip from text (Wan, async)
