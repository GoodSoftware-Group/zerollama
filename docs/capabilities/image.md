---
title: Image
description: MLX image generation fast path and ComfyUI workflow path.
---

**Status:** `beta`

Image generation is a working surface in zerollama, with **two main backend styles** today:

- **MLX imagegen fast path** for the built-in path on supported models
- **ComfyUI utility path** for richer workflows such as edits, ControlNet, or heavier model families

## API surface

- `/api/generate` for image-capable models on the zerollama side
- OpenAI-style `/v1/images/*` where the backend path supports it

## Current backends

| Backend | Status | What it is good at | Deep docs |
|---|---|---|---|
| **MLX imagegen** | `beta` | Fast path for Z-Image Turbo / MLX-shaped image generation | [imagegen-zimage-turbo.md](../design/media/imagegen-zimage-turbo.md) |
| **ComfyUI** | `beta` | Workflow-heavy utility path: instruction edit, img2img, ControlNet, LoRA, bigger model families | [comfyui-image-backend.md](../design/media/comfyui-image-backend.md) |
| **external-image** | `lab` / escape hatch | One-off subprocess integration, not the preferred product path | [multimodal-backends.md](../design/media/multimodal-backends.md) |

## What is stable at the capability layer

- Zerollama owns the **agent-facing API**, scheduling, and VRAM handoff.
- Backend-specific complexity stays behind manifests / workflow bindings where possible.
- Image generation is separate from chat and should be treated as its own capability family.

## Current guidance

- Prefer the **MLX fast path** when the model family already fits that shipped path.
- Prefer **ComfyUI** when you need utility workflows or model families zerollama should not reimplement natively.
- Treat `external-image` as an escape hatch, not the main documented surface.

## Not this capability

- **Not video generation**: use [video.md](video.md).
- **Not multimodal chat understanding**: image understanding remains part of the multimodal/chat surface, not this page.

## Related docs

- Product backlog: [media roadmap](../roadmaps/media.md#image-generation--mlx-fast-path--comfyui-utility-partial)
- Backend overview: [multimodal-backends.md](../design/media/multimodal-backends.md)
