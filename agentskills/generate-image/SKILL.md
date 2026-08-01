---
name: generate-image
description: "Generate or edit images via a zerollama server (MLX fast path or ComfyUI backend)."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, image, imagegen, comfyui, mlx, text-to-image, img2img]
    category: mlops
    related_skills: [zerollama-integration, download-model]
---

# Generate Image Skill

Generate or edit images through a [zerollama](https://github.com/GoodSoftware-Group/zerollama)
server. Zerollama routes image requests to one of two backends depending on
the model's manifest: a native **MLX pipeline** (fast, seconds on a warm GPU
Mac) or a **ComfyUI** bridge (heavier DiT families with LoRA/ControlNet
support). Agents use the same API either way.

## Compatibility check

This skill targets zerollama **tip/dev**, not a specific pinned
release — not every server will have every endpoint/flag below yet.
Verify before relying on this in an unattended flow, especially
against a host you don't control:

```bash
zerollama --version                      # binary build
curl -s http://localhost:11434/api/version | jq   # server build (if reachable)
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/tags   # 200/400 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/api/generate -d '{}'   # 400/422 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/v1/images/generations -d '{}'   # 400/422 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/v1/images/edits -d '{}'   # 400/422 = route exists; 404 = missing on this build
```

A **404** on an endpoint above (or an unrecognized flag/subcommand) means this build predates the feature this skill
describes — check [`CHANGELOG.md`](../CHANGELOG.md) for when it
landed, or upgrade (`git pull && ./scripts/build/build_zerollama_mac.sh`)
rather than assuming the request shape is wrong.

## When to Use

- The user asks to generate, create, or edit an image locally (not via a
  cloud image API)
- Choosing which local image model/workflow fits speed vs. quality needs
- Debugging an image request that returns an error or times out

## Prerequisites

- zerollama server running (default `http://localhost:11434`)
- An image-capable model pulled/registered (`GET /api/tags`, filter for
  `comfy/*` or MLX image models like `z-image-turbo`, `flux2-klein`)
- For `comfy/*` models: a running ComfyUI instance the server can reach
  (`OLLAMA_COMFYUI_URL`, default `http://127.0.0.1:8188`) — see
  `docs/comfyui-image-backend.md` in the zerollama repo

## API Contract

Three equivalent surfaces, all resolved by `handleImageGenerate`:

| Endpoint | Shape | Notes |
|---|---|---|
| `POST /api/generate` | native | `model` + `prompt`; image bytes come back base64 in the response |
| `POST /v1/images/generations` | OpenAI-compatible | JSON body, same fields as OpenAI's image API |
| `POST /v1/images/edits` | OpenAI-compatible | `multipart/form-data` with an input `image` for img2img/instruction-edit |
| `GET /api/image/workflows?model=<name>` | discovery | lists named workflows (`t2i`, `img2img`, `edit`, `controlnet`, …) and required fields for a `comfy/*` model before you guess |

## How to Run

```bash
# 1. Discover image-capable models
curl -s http://localhost:11434/api/tags | grep -i -E 'comfy|image|flux|z-image'

# 2. (comfy/* models only) discover the model's named workflows
curl -s "http://localhost:11434/api/image/workflows?model=comfy/qwen-image"

# 3. Generate (OpenAI-compatible)
curl -s -X POST http://localhost:11434/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "z-image-turbo",
    "prompt": "a watercolor fox in a forest",
    "size": "1024x1024"
  }'

# 4. Edit / img2img (comfy/* models with an "edit" or "img2img" workflow)
curl -s -X POST http://localhost:11434/v1/images/edits \
  -F model="comfy/qwen-image-edit" \
  -F image=@input.png \
  -F prompt="make the sky sunset orange" \
  -F 'options={"workflow":"edit"}'
```

## Choosing a Model

| Need | Model family | Backend |
|---|---|---|
| Fast interactive gen (seconds, Mac GPU) | `z-image-turbo`, `flux2-klein-9b` | MLX (`x/imagegen`) |
| Instruction edit / img2img / ControlNet / LoRA | `comfy/qwen-image`, `comfy/qwen-image-edit`, `comfy/flux1-dev`, `comfy/flux2-dev`, `comfy/glm-image` | ComfyUI bridge |
| Heavy DiT models (multi-minute on 16GB) | `comfy/flux2-dev`, `comfy/glm-image` | ComfyUI, treat as async |

Prefer the MLX fast path (`z-image-turbo`, `flux2-klein-9b`) for interactive
agent use. Only reach for `comfy/*` when you need edit/ControlNet/LoRA that
MLX doesn't support.

## Pitfalls

- **404/empty on `/api/image/workflows`** — the model isn't registered as a
  `comfy/*` manifest, or `OLLAMA_COMFYUI_WORKFLOWS_ROOT` isn't set when
  `zerollama serve` doesn't start from the repo root.
- **Comfy job hangs/times out** — `OLLAMA_COMFYUI_TIMEOUT` defaults to 10m;
  Qwen/FLUX.2/GLM on 16GB can legitimately take minutes. Treat as best-effort
  async work, not an interactive call.
- **VRAM contention** — image generation evicts ggml/runtime chat models
  first (`vram.PrepareForImageGen`); expect a brief pause if a chat model was
  loaded.
- **Workflow JSON node-id mismatch** — shipped `scripts/comfyui/*/*.json`
  graphs are worked examples, not guaranteed drop-in; Comfy custom-node class
  names and checkpoint filenames drift per install.

## Related

- `zerollama-integration` — generic API contract, sizing, pitfalls
- `download-model` — pulling/registering an image model before use
