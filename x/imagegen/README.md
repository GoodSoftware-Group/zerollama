# `x/imagegen` — MLX diffusion runner

Go package that implements Ollama-compatible **image generation** via an MLX subprocess. Used for models with the `image` capability (e.g. `x/z-image-turbo`).

**Operator guide:** [docs/imagegen-zimage-turbo.md](../../docs/imagegen-zimage-turbo.md)

---

## Why a subprocess

- MLX-C requires a dedicated thread and GPU context; mixing with Go's ggml CGO and embedded Python in one process risks allocator and driver contention.
- Crash isolation: a bad `mlx_eval` in denoise does not take down the whole daemon.
- Same pattern as upstream Ollama MLX imagegen — we extend it for **CUDA** and **tight VRAM** hosts.

The Go `server.Scheduler` treats `imagegen.NewServer()` as `llm.LlamaServer`: load, ping, completion, close.

---

## Package layout

| Path | Purpose |
|------|---------|
| `server.go` | Subprocess lifecycle, HTTP client to runner, `Completion` NDJSON parse |
| `runner.go` | MLX runner HTTP server (`/completion`, `/health`) |
| `imagegen.go` | Request handling, progress streaming, PNG base64 |
| `cli.go` | `zerollama run` client-side error surfacing |
| `models/zimage/` | Z-Image turbo pipeline (text encoder, transformer, VAE) |
| `manifest/` | Ollama blob → safetensors mmap loader |
| `mlx/` | Go bindings to MLX-C (eval, memory, arrays) |
| `size/` | Aspect presets + VRAM-aware max side |
| `decode_latents.go` | CLI entry for CPU VAE subprocess |
| `latents/` | Latent tensor file format for subprocess handoff |

---

## Request flow

1. Go `scheduleRunner` starts or reuses MLX runner for model key.
2. Runner loads manifest on first request (`loadImageModel`).
3. `handleImageCompletion` holds `imageGenMu` — **one generation at a time per process** (WHY: peak VRAM is already at the card limit).
4. `generateOnMLXThread` resolves dimensions, calls `GenerateImage`, encodes PNG.
5. Z-Image `generate()` stages: encoder → free → transformer → denoise → export latents → subprocess VAE.

---

## VRAM conventions (CUDA 16GB)

Documented in depth in the operator guide. Summary for contributors:

- **Defer** text encoder load until first `generate` when `mlx.GPUIsAvailable()`.
- **Batch** weight materialization via `manifest/weights.go` → `mlx.EvalErrBatched(16, ...)`.
- **Keep** mmap `SafetensorsFile` handles in `nativeCache` until `ReleaseAll()` — freeing after eval invalidates CUDA buffers.
- **Free** text encoder before transformer load; call `mlx.ResumeCleanup()` to drop graph debris.
- **Keep** transformer weights between requests on CUDA (`freeTransformerWeights` no-op release).
- **Decode** VAE in a **fresh CPU subprocess** after denoise on CUDA.

---

## Building / testing

Requires MLX-C built and installed — see [docs/imagegen-zimage-turbo.md](../../docs/imagegen-zimage-turbo.md#one-time-build-5080--sm_120).

```bash
# From repo root — links imagegen into zerollama binary
CGO_ENABLED=1 go build -o zerollama .

# Unit tests (no GPU required for size/manifest)
go test ./x/imagegen/size/... ./x/imagegen/manifest/...
```

`x/imagegen/nn` tests require MLX shared libs and may crash in CI without GPU — pre-existing.

---

## Adding a model family

1. Implement `ImageModel` (`GenerateImage` in `imagegen.go`).
2. Register in `loadImageModel` switch (`DetectModelType`).
3. Ensure manifest layout matches `manifest.LoadManifest` component prefixes.
4. If VRAM > 16GB at full resolution, add staged loading like `zimage.Model`.
