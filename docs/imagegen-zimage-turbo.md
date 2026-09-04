# MLX image generation — Z-Image Turbo (`x/z-image-turbo`)

**Audience:** Operators and developers running diffusion image models on NVIDIA CUDA (especially 16 GB cards like RTX 5080) or Apple Metal.

**Related:** [gpu-5080-operator-guide.md](./gpu-5080-operator-guide.md) (serve env + gate sequence), [scheduling-vram-policy.md](./scheduling-vram-policy.md) (Phase 8 VRAM broker), [ROADMAP.md](./ROADMAP.md).

---

## Why this path exists

Zerollama already runs **three** inference stacks on one GPU:

1. **Go ggml runners** — legacy chat/completion for some model families.
2. **Python runtime** — default text path (`llama-server` subprocess on `:8081`).
3. **Embedded training** — same Go process, separate job queue.

Diffusion models (`x/z-image-turbo`, Flux variants) use a **fourth** stack: an **MLX imagegen subprocess** linked against `libmlxc.so` (MLX-C CUDA or Metal). **Why not ggml or the Python runtime?**

- Z-Image is a **manifest of safetensor blobs** loaded through MLX, not GGUF.
- Denoising needs a full transformer + text encoder + VAE pipeline with staged VRAM — different from llama.cpp KV sizing.
- Upstream Ollama ships MLX imagegen on macOS; this fork adds **CUDA** support and **16 GB VRAM** survival tactics.

The Go daemon still owns scheduling, HTTP, and VRAM handoff; only the compute runs in the MLX runner child process.

---

## Architecture

```text
Client                    Go (:8080)                         MLX runner (child)
  |  zerollama run x/z-image-turbo "prompt"
  |------------------------>|  handleImageGenerate
  |                         |  PrepareForImageGen → evict ggml + runtime
  |                         |  scheduleRunner → imagegen.NewServer
  |                         |  HTTP POST /completion (NDJSON stream)
  |                         |---------------------------------->|
  |                         |                                   | loadImageModel (manifest)
  |                         |                                   | text encode → denoise → VAE
  |  progress steps         |<-------- step/total --------------|
  |  final PNG (base64)     |<-------- image + done -------------|
```

**Why dimension resolution lives in the runner, not Go:** The serve process may not have MLX loaded, so `mlx.GPUIsAvailable()` is false in Go even when the child sees CUDA. Pre-resolving width/height in `routes.go` with the wrong `maxSide` caused a **double clamp** (e.g. 1024² requested → 384² effective without the client knowing). Go now only validates `aspect_ratio` strings; the runner calls `size.Resolve` with the correct GPU context.

---

## VRAM strategy on 16 GB (WHY each stage)

The Z-Image turbo manifest is ~12 GB of weights. A 16 GB card cannot hold text encoder + transformer + VAE + activations at once. The pipeline **serializes** components:

| Stage | What loads | Why |
|-------|------------|-----|
| **Serve / first load** | Tokenizer only; text encoder **deferred** on CUDA | Avoid holding ~4.5 GB encoder while chat models may still be loaded. |
| **First `generate`** | Text encoder → encode prompt → **free encoder** | Encoder VRAM must drop before transformer materialization. |
| **Transformer load** | Batched `mlx_eval` (16 tensors/chunk) | Single eval over all weights spikes VRAM and OOMs after encoder release. |
| **Denoise** | Transformer resident; TeaCache **off** on CUDA | CUDA graphs disabled; TeaCache reuse is unreliable without compile. |
| **Post-denoise** | Export latents to disk; **keep transformer resident** on CUDA | Reloading transformer between requests leaked handles and OOM'd the second request. |
| **VAE decode** | **Fresh CPU subprocess** (`decode_latents`) | In-process CUDA VAE after denoise hit allocator heap corruption / OOM on 5080. |

Default output size on CUDA hosts: **384×384** long edge (`size.MaxSide(true)`). Override with `ZEROLLAMA_IMAGE_MAX_SIDE` or explicit `width`/`height` in the API (still clamped to max side).

---

## One-time build (5080 / sm_120)

**Why a separate CMake tree:** MLX-C uses its own CUDA backend (`mlx_cuda_v12`), not ggml's `cuda_v12` runners. Both must be on `LD_LIBRARY_PATH` / `OLLAMA_LIBRARY_PATH`.

```bash
cd /root/zerollama
apt install -y libopenblas-dev liblapacke-dev
export PATH=/usr/local/cuda-12.8/bin:$PATH   # sm_120 needs CUDA 12.8+ nvcc

cmake -B build-mlx --preset "MLX CUDA 12" \
  -DMLX_CUDA_ARCHITECTURES=120-real \
  -DBLAS_INCLUDE_DIRS=/usr/include/x86_64-linux-gnu \
  -DLAPACK_INCLUDE_DIRS=/usr/include
cmake --build build-mlx --target mlx --target mlxc --parallel

# Optional but recommended on 16GB: patch allocator before build
./scripts/mlx/patch_mlx_cuda_vram.sh
cmake --build build-mlx --target mlx --target mlxc --parallel

cmake --install build-mlx --component MLX --strip
sudo mkdir -p /usr/lib/ollama/mlx_cuda_v12
sudo cp -a dist/lib/ollama/mlx_cuda_v12/* /usr/lib/ollama/mlx_cuda_v12/
```

**Why three patch scripts (apply in order):**

```bash
# 1. mlx-c/array.cpp — add mlx_array_detach + mlx_go_export_latents_bin_d2h
./scripts/mlx/patch_mlx_c_array.sh

# 2. (first build only / clean tree) — strip debug fprintfs added during OOM
#    diagnosis; safe no-op if already clean
./scripts/mlx/patch_mlx_c_debug_cleanup.sh

# 3. mlx/backend/cuda/allocator.cpp — cudaMalloc, 90% limit, disable recycle
./scripts/mlx/patch_mlx_cuda_vram.sh

cmake --build build-mlx --target mlx --target mlxc --parallel
```

| Script | File patched | Why |
|--------|-------------|-----|
| `patch_mlx_c_array.sh` | `mlx-c-src/mlx/c/array.cpp` | Adds `mlx_array_detach` (break graph links before free) and `mlx_go_export_latents_bin_d2h` (direct `cudaMemcpy` D2H for latent export after denoise — bypasses `mlx::core::copy` which faults post-checkpoint on CUDA) |
| `patch_mlx_c_debug_cleanup.sh` | same file | Strips `fprintf` debug instrumentation added during OOM diagnosis; not needed in production but must be removed before shipping |
| `patch_mlx_cuda_vram.sh` | `mlx-src/mlx/backend/cuda/allocator.cpp` | Force sync `cudaMalloc`/`cudaFree` (async pools reserve VA that counts as VRAM); 90% memory limit; disable buffer-cache recycle (heap corruption after checkpoint frees). Helper: `_patch_mlx_cuda_vram.py`. |

All three scripts are idempotent — safe to re-run if you're unsure whether a patch is already applied.

---

## Serve and run

Use `scripts/serve/serve_gpu_example.sh` as a template. Minimum for imagegen:

```bash
export OLLAMA_LIBRARY_PATH=/usr/lib/ollama:/usr/lib/ollama/cuda_v12:/usr/lib/ollama/mlx_cuda_v12
export LD_LIBRARY_PATH=/usr/lib/ollama/mlx_cuda_v12:$LD_LIBRARY_PATH
export OLLAMA_MAX_LOADED_MODELS=1   # WHY: imagegen needs the full card

zerollama serve
```

```bash
zerollama pull x/z-image-turbo          # registry name — not library/z-image
zerollama stop <other-model>            # free VRAM (~12 GB for weights alone)

OLLAMA_HOST=127.0.0.1:8080 zerollama run x/z-image-turbo "a sunset over mountains"
# Saves PNG; default ~384×384 on 16GB CUDA
```

**API** (same as `zerollama run`):

```bash
curl -s http://127.0.0.1:8080/api/generate -d '{
  "model": "x/z-image-turbo",
  "prompt": "a sunset over mountains",
  "stream": true,
  "aspect_ratio": "16:9"
}'
```

NDJSON lines include `completed`/`total` during denoise and `image` (base64 PNG) on the final line.

---

## Environment reference

| Variable | Default (CUDA 16GB) | Why |
|----------|-------------------|-----|
| `OLLAMA_LIBRARY_PATH` | — | Must include `mlx_cuda_v12` or runner falls back to CPU. |
| `OLLAMA_MAX_LOADED_MODELS` | `1` recommended | Image model + chat runner exceeds 16 GB. |
| `ZEROLLAMA_IMAGE_MAX_SIDE` | `384` when GPU/env hints CUDA | Long-edge cap; diffusion activations scale with pixels². |
| `CUDA_VISIBLE_DEVICES` | — | Unset in serve; set in runner. Used as CUDA hint for `MaxSide` when parent has no MLX. |

---

## Troubleshooting

### `mlx eval failed (ret=1)` / reload transformer

**Cause:** GPU OOM during weight materialization — often second request after encoder/transformer churn, or chat model still loaded.

**Fix:**

1. `zerollama ps` → `zerollama stop` other models.
2. Confirm `PrepareForImageGen` ran (logs: evict ggml + runtime handoff).
3. Rebuild MLX with `patch_mlx_cuda_vram.sh` if not already.
4. Lower `ZEROLLAMA_IMAGE_MAX_SIDE` (e.g. `320`).

### `image generation completed without image data`

**Cause:** Stream ended with `done: true` but no `image` field — usually the MLX subprocess failed after streaming started, or the scheduler killed the runner mid-request.

**Fix:** Check serve logs for the real error (`error: ...` in NDJSON). Recent fixes:

- Scheduler **defers** unload of in-use imagegen runners during `UnloadAllRunners` (training handoff no longer tears down active streams).
- Go writes `error: ...` on the NDJSON stream when failure happens after headers commit.

### Scheduler panic on load failure

**Cause:** `activeLoading` cleared by concurrent VRAM handoff while `load()` still held a local handle → nil deref on error path.

**Fix:** `clearActiveLoading(llama)` closes either `activeLoading` or the local handle — shipped in `server/sched.go`.

### `/info` panic or CPU-only ggml on 5080

**Cause:** Dereferencing invalid `ggml_backend_dev_props` string pointers on newer drivers.

**Fix:** Use `ggml_backend_dev_name` / `ggml_backend_dev_description` C APIs directly (`ml/backend/ggml/ggml.go`).

### Wrong resolution (expected 384, got 1024 request path)

**Cause:** Go pre-resolved dimensions before subprocess GPU detection.

**Fix:** Resolution only in runner + `size` package; Go validates aspect presets only.

---

## Code map

| Piece | Path | Role |
|-------|------|------|
| HTTP entry | `server/routes.go` (`handleImageGenerate`) | VRAM prep, schedule runner, stream NDJSON |
| VRAM broker | `server/vram/broker.go` (`PrepareForImageGen`) | Evict other runners + runtime sidecar |
| Scheduler | `server/sched.go` | Load/evict imagegen; defer in-use unloads |
| Go ↔ subprocess | `x/imagegen/server.go`, `runner.go` | `llm.LlamaServer` adapter |
| HTTP handler | `x/imagegen/imagegen.go` | Progress streaming, base64 encode |
| Z-Image model | `x/imagegen/models/zimage/zimage.go` | Staged load, denoise, latent export, CPU VAE subprocess |
| Dimensions | `x/imagegen/size/size.go` | `MaxSide`, aspect presets, clamp |
| Weight I/O | `x/imagegen/manifest/weights.go` | mmap safetensors, batched eval |
| MLX bindings | `x/imagegen/mlx/mlx.go` | `EvalErrBatched`, VRAM trim, cleanup |
| CPU VAE helper | `x/imagegen/decode_latents.go` | Subprocess entry for post-denoise decode |
| MLX allocator patch | `scripts/mlx/patch_mlx_cuda_vram.sh` | `cudaMalloc`, 90% limit, disable recycle |
| MLX-C array patch | `scripts/mlx/patch_mlx_c_array.sh` | `mlx_array_detach` + `mlx_go_export_latents_bin_d2h` |
| MLX-C debug cleanup | `scripts/mlx/patch_mlx_c_debug_cleanup.sh` | Strip debug `fprintf`s before production build |
| Serve template | `scripts/serve/serve_gpu_example.sh` | Library paths + comments |

Developer notes: [x/imagegen/README.md](../x/imagegen/README.md).

---

## Limitations (Jun 2026)

- **384 default** on 16 GB CUDA — quality/speed tradeoff, not a hard model limit; raise `ZEROLLAMA_IMAGE_MAX_SIDE` only with headroom.
- **VAE on CPU subprocess** — extra latency vs in-process GPU decode; avoids CUDA heap bugs after denoise.
- **Transformer kept resident** between CUDA requests — faster repeat prompts, higher idle VRAM until `keep_alive` expires.
- **TeaCache disabled on CUDA** — enable only after CUDA graph/compile story is stable.
- **Experimental** — not part of `gpu_5080_session.sh` gate yet; manual smoke: pull + `zerollama run` + verify PNG.
