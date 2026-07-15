# `server/modality/comfyui`

Go HTTP bridge for `modality_backends.image=comfyui`.

**Operator / agent guide (WHYs):** [docs/comfyui-image-backend.md](../../../docs/comfyui-image-backend.md)

## Why this package

- Agents need edit / ControlNet / LoRA on many HF DiTs; MLX only covers Z-Image + Klein 4B.
- ComfyUI already packages those graphs — Zerollama **orchestrates**, it does not reimplement DiTs.
- Keep `/api/generate` and `/v1/images/*` unchanged so agents do not learn a third API.

## Layout

| File | Role |
|------|------|
| `client.go` | Comfy `/upload/image`, `/prompt`, `/history/{id}`, `/view`; execution_error parsing |
| `workflow.go` | Template load + bindings → node input injection (deep-copy per request) |
| `backend.go` | `Generate` / `ListWorkflows` + `resolveWorkflowDir` |

## Tests

```bash
go test ./server/modality/comfyui/...
# optional live Comfy:
RUN_E2E_COMFY=1 COMFY_E2E_WORKFLOW_DIR=/path/to/workflows go test ./server/modality/comfyui/... -run TestE2ESmoke -v
```
