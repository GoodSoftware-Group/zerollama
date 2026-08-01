---
name: download-model
description: "Pull, register, list, and remove models on a zerollama server (GGUF chat models, image/video/audio backends)."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, ollama, pull, model-management, registry, comfyui, wan]
    category: mlops
    related_skills: [zerollama-integration, generate-image, generate-video, doctor-model, model-suggester]
---

# Download Model Skill

Pull, register, inspect, and remove models on a [zerollama](https://github.com/GoodSoftware-Group/zerollama)
server. Most chat/embedding GGUF models are a normal `ollama`-style pull;
image (`comfy/*`) and video (Wan) models use **config-only manifests** that
point at weights living outside Ollama's blob store, so "downloading" them
is a two-step install + register instead of a single pull.

## When to Use

- The user asks to download, pull, or install a model for local inference
- Registering an image (ComfyUI) or video (Wan) model that isn't a plain
  GGUF pull
- Checking whether a model exists locally before running/generating with it
- Freeing disk space by removing an unused model

## GGUF chat / embedding models (standard pull)

```bash
# Pull (streams NDJSON progress by default)
curl -s http://localhost:11434/api/pull -d '{"model": "gemma3"}'

# Pull without streaming (single final JSON response)
curl -s http://localhost:11434/api/pull -d '{"model": "gemma3", "stream": false}'

# List installed models
curl -s http://localhost:11434/api/tags

# Inspect a model's manifest (add "verbose": true for full details)
curl -s http://localhost:11434/api/show -d '{"model": "gemma3"}'

# Pre-flight VRAM check before pulling something huge
curl -s -X POST http://localhost:11434/api/can-load \
  -H 'Content-Type: application/json' -d '{"model":"gemma3"}'

# Remove a model
curl -s -X DELETE http://localhost:11434/api/delete -d '{"model": "gemma3"}'
```

`insecure: true` in the pull/push body allows an insecure registry
connection; leave it unset unless the user explicitly needs it.

CLI equivalents work the same way: `zerollama pull gemma3`,
`zerollama list`, `zerollama rm gemma3`.

## Image models (`comfy/*`)

ComfyUI-backed image models are **config-only** — weights live in ComfyUI's
own `models/` tree, not Ollama blobs. There is nothing to "pull" from the
registry; register the manifest instead:

```bash
# Register all shipped presets
./scripts/register_comfy_models.sh

# Or a single model
./scripts/register_comfy_models.sh comfy/qwen-image modelfiles/comfy-qwen-image/config.json
```

Then confirm it shows up: `curl -s http://localhost:11434/api/tags | grep comfy/`.
Actual model weights must be downloaded into ComfyUI's `models/` directory
separately (ComfyUI-GGUF for quantized weights on 16GB cards) — zerollama
does not manage or blob those files. See `docs/comfyui-image-backend.md`.

MLX-native image models (`z-image-turbo`, `flux2-klein-9b`) are regular
pulls like any chat model: `curl -s http://localhost:11434/api/pull -d '{"model":"z-image-turbo"}'`.

## Video models (Wan)

Also config-only; install the third-party checkpoint + register in one pass:

```bash
./scripts/video/install_wan_video.sh --profile all   # or 1.3b | 2.2
./scripts/video/register_wan_models.sh
```

This downloads Wan checkpoints into `~/.zerollama/third_party/wan/` and
registers `wan2.1-t2v-1.3b` / `wan2.2-ti2v-5b` manifests. Confirm with
`curl -s http://localhost:11434/api/tags | grep wan`.

## Choosing the right path

| Model type | "Download" mechanism |
|---|---|
| GGUF chat / embedding | `POST /api/pull` (registry blob download) |
| MLX image (`z-image-turbo`, `flux2-klein-9b`) | `POST /api/pull` |
| ComfyUI image (`comfy/*`) | `scripts/register_comfy_models.sh` + separate ComfyUI weight download |
| Wan video (`wan2.*`) | `scripts/video/install_wan_video.sh` + `register_wan_models.sh` |

## Pitfalls

- **Don't assume `/api/pull` works for `comfy/*` or `wan*` models** — these
  are config-only manifests; a plain pull will fail or no-op because there's
  no GGUF blob to fetch.
- **Disk/VRAM before pulling large models** — check `/api/can-load` first
  for chat models; for image/video, confirm host disk space since Wan/FLUX.2
  checkpoints are multi-GB.
- **Streaming vs. non-streaming pulls** — default `stream: true` emits
  NDJSON progress lines; set `stream: false` if the caller just wants a
  final success/failure JSON.
- **Name must include the tag** — `qwen3-coder-next:6bit`, not
  `qwen3-coder-next`. Confirm exact names against `GET /api/tags` before
  generating/chatting with a model.

## Related

- `zerollama-integration` — generic API contract, sizing, pitfalls
- `generate-image` — using an image model once it's registered
- `doctor-model` — verify a model's manifest/blob health after pulling
- `generate-video` — using a Wan video model once it's registered
- `model-suggester` — deciding which model to pull in the first place
