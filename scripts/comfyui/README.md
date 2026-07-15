# ComfyUI workflow templates (worked examples)

API-format ComfyUI graphs + Zerollama **bindings** for `modality_backends.image=comfyui`.

**Full guide:** [docs/comfyui-image-backend.md](../../docs/comfyui-image-backend.md)

## Why these files exist

Agents pass `options.workflow=t2i|edit|img2img|controlnet` without speaking Comfy node ids. Each `<name>.json` pairs:

- `graph` — Comfy `POST /prompt` payload shape (`class_type` + `inputs`)
- `bindings` — logical fields (`prompt`, `seed`, …) → `{node_id, field}`
- `requires` — fields the renderer rejects if missing

**Why “worked examples” not “verified goldens”:** custom-node class names and checkpoint filenames differ per install. Calibrate in Comfy’s UI before production. Once the graph runs, only adjust bindings/filenames — no Go changes.

## Layout

```
scripts/comfyui/
  qwen-image/          t2i, img2img, controlnet
  qwen-image-edit/     edit (prompt binds to TextEncodeQwenImageEdit)
  flux1-dev/           t2i
  flux2-dev/           t2i
  glm-image/           t2i (heavy on 16GB)
  flux2-klein-9b/      t2i (MLX Klein stays 4B-shaped)
```

Register manifests: `../../scripts/register_comfy_models.sh`.

## Rules of thumb

1. **No fake LoRA defaults** (`none.safetensors`) — Comfy fails load; omit the LoRA node unless a real file is intended (edit Lightning LoRA).
2. Bind agent `prompt` to the node that actually feeds the sampler’s positive path.
3. Point `backend_paths.comfy_workflow_dir` at this directory (relative + `OLLAMA_COMFYUI_WORKFLOWS_ROOT`, or absolute/`~/`).
