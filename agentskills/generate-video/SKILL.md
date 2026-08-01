---
name: generate-video
description: "Generate text-to-video clips via a zerollama server's async OpenAI-compatible Videos API (Wan)."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, video, text-to-video, wan, async-job]
    category: mlops
    related_skills: [zerollama-integration, download-model]
---

# Generate Video Skill

Generate text-to-video (T2V) clips through a [zerollama](https://github.com/GoodSoftware-Group/zerollama)
server using the OpenAI-compatible, **async** Videos API. Generation runs as
a queued job on the embedded training worker (Wan), not as a synchronous
chat-style call — a single video can take 30–60+ minutes, so agents must
poll for completion.

## When to Use

- The user asks to generate a video clip locally from a text prompt
- Polling/checking status of a previously submitted video job
- Debugging a `503` from `POST /v1/videos` or a stuck/deferred job

## Prerequisites

- zerollama server running with `OLLAMA_TRAINING=true` (default when the
  embedded PyTorch worker is available) — otherwise `POST /v1/videos`
  returns `503`
- GPU with ~16GB for the shipped presets (`wan2.1-t2v-1.3b`, `wan2.2-ti2v-5b`)
- Wan repo + checkpoints installed and registered (see Install below)

## API Contract

| Endpoint | Method | Notes |
|---|---|---|
| `/v1/videos` | `POST` | Body: `{model, prompt, size?, seconds?, seed?, options?}`. Returns a job id immediately (may be `defer-*` if the fleet is busy with inference). |
| `/v1/videos/:id` | `GET` | Poll job status/progress. |
| `/v1/videos/:id/content` | `GET` | Download the finished `video/mp4`; **404/425 until `status` is `completed`.** |

`options` is backend-specific (e.g. `frames`, `steps` for Wan). `seconds` is
accepted for OpenAI compatibility but actual frame count is driven by
`options`.

## How to Run

```bash
# 1. Submit a job
JOB=$(curl -s -X POST http://localhost:11434/v1/videos \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "wan2.2-ti2v-5b",
    "prompt": "a paper boat sailing down a rain-soaked street, cinematic",
    "size": "1280x704"
  }')
echo "$JOB"
ID=$(echo "$JOB" | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')

# 2. Poll until status == "completed" (can take tens of minutes)
watch -n 15 "curl -s http://localhost:11434/v1/videos/$ID"

# 3. Download the finished clip
curl -s http://localhost:11434/v1/videos/$ID/content -o out.mp4
```

## Install (one-time, before first use)

```bash
./scripts/video/install_wan_video.sh --profile all   # or 1.3b | 2.2
./scripts/video/register_wan_models.sh
```

Checkpoints live under `~/.zerollama/third_party/wan/` — config-only
manifests, not Ollama blobs.

## Pitfalls

- **Never poll synchronously in a blocking loop with short sleeps** — jobs
  run for tens of minutes; poll every 15–30s and yield between checks.
- **`503` on submit** — training isn't embedded (`OLLAMA_TRAINING=false` or
  PyTorch missing). This is a server config issue, not a request issue.
- **`defer-*` job ids** — when the fleet is busy with inference, the job is
  queued behind chat traffic instead of failing with `409`. Keep polling the
  **same** id after promotion; do not resubmit.
- **Content 404 before completion** — `GET /v1/videos/:id/content` only
  serves the file once `status` is `completed`; check status first.
- **16GB host RAM tight** — `WAN_UNLOAD_T5=1` (default on 16g presets)
  releases the T5 text encoder after encode to fit in 16GB RAM.
- **GPU contention with chat/training** — video generation shares the GPU
  with ggml/runtime inference and the training queue via the VRAM broker;
  expect chat models to be evicted while a video job runs.

## Related

- `zerollama-integration` — generic API contract, sizing, pitfalls
- `download-model` — installing/registering Wan video models
