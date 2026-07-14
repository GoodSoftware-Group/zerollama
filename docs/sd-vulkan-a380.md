# Stable Diffusion on Intel Arc A380 (Vulkan)

**Audience:** Operators running **text-to-image** locally on 6 GB Arc via **stable-diffusion.cpp** and zerollama’s `external-image` backend.

**Related:** [multimodal-backends.md](./multimodal-backends.md), [a380-runbook.md](./a380-runbook.md), [bench-cache.md](./bench-cache.md)

---

## Why this path

Zerollama’s built-in **MLX imagegen** (Z-Image, Flux) targets **16 GB CUDA/Metal**. On A380 you want:

- **stable-diffusion.cpp** with **Vulkan** (same ggml family as llama.cpp) — **why:** one GPU stack, one operator toolchain, weights as GGUF beside chat models
- **SD 1.5 Q4** (~2 GB weights) — fits 6 GB with `--vae-on-cpu`
- **`modality_backends.image=external-image`** — subprocess hook already in zerollama; per-model weights via manifest `backend_paths` — **why:** diffusion UNet is not loaded into the chat ggml runner; subprocess keeps scheduler and VRAM accounting simple

OpenVINO integration (second stack, same API): see [sd-openvino-a380.md](./sd-openvino-a380.md).

---

## Install (one time)

```bash
cd ~/zerollama
chmod +x scripts/image/install_stable_diffusion.sh scripts/image/sd_external_image.sh scripts/image/register_sd_models.sh

./scripts/image/install_stable_diffusion.sh          # prebuilt Vulkan binary + SD1.5 Q4_0 GGUF
./scripts/image/register_sd_models.sh
```

Install layout:

| Path | Purpose |
|------|---------|
| `~/.zerollama/third_party/sd-cpp/bin/sd-cli` | stable-diffusion.cpp Vulkan binary |
| `~/.zerollama/third_party/sd-cpp/models/stable-diffusion-v1-5-Q4_0.gguf` | SD 1.5 weights |
| `scripts/image/sd_external_image.sh` | zerollama wrapper |

---

## Service env

Add to `/etc/zerollama/a380-llama.env` (or export before `zerollama serve`):

```bash
OLLAMA_EXTERNAL_IMAGE_BIN=/root/zerollama/scripts/image/sd_external_image.sh
```

Restart `zerollama.service` after changes.

---

## Models (local)

| Tag | Weights | Notes |
|-----|---------|--------|
| `sd15-vulkan` | SD 1.5 Q4_0 | Default baseline |
| `sd15-q8-vulkan` | SD 1.5 Q8_0 | Sharper, +~130 MB vs Q4 |
| `sd15-turbo-vulkan` | SD-Turbo Q8 | **Fast** — 4 steps, cfg 1.0 |
| `sdxl-vulkan` | SDXL Q4_0 | **Experimental** — 768², vae tiling |

Install all: `./scripts/image/install_stable_diffusion.sh --models-only`

---

```bash
zerollama run sd15-vulkan "a watercolor lighthouse at sunset"
```

API (`POST /api/generate`):

```json
{
  "model": "sd15-vulkan",
  "prompt": "a red cube on a wooden table",
  "width": 512,
  "height": 512,
  "stream": false
}
```

Response includes base64 PNG in the `image` field.

---

## Bench and list

**Why bench image tags:** PERF in `zerollama ls` is the fastest way to compare Q4 vs Q8 vs turbo vs OpenVINO on *this* box.

```bash
zerollama bench sd15 --force --epochs 1 --warmup 0
zerollama ls sd15              # PERF column: e.g. 35s for sd15-turbo-vulkan
zerollama ls image             # all image-capable tags (local + cloud routes)
```

See [bench-cache.md](./bench-cache.md) — image bench uses `TotalDuration`, caps at 2 timed epochs, and stores `kind: image` in `~/.ollama/bench.json`.

---

## A380 / Intel Vulkan notes

| Flag | Default in manifest | Why |
|------|---------------------|-----|
| `diffusion_fa` | `true` | Required on Intel Mesa ANV — without it output is noise |
| `vae_on_cpu` | `true` | Keeps VAE off 6 GB VRAM during decode |
| `vae_tiling` | `false` | Enable if VAE still OOMs |
| Resolution | 512×512 | Safe default; 768 may OOM |

Tune in `modelfiles/sd15-vulkan/config.json` under `image_generation`.

---

## VRAM / scheduling

- Manifest sets `concurrency_groups: ["imagegen"]` so schedulers can evict chat models before image jobs (same pattern as MLX imagegen).
- Unload chat models before heavy image runs on 6 GB: `zerollama stop gemma4-e2b-qat-xl`.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `OLLAMA_EXTERNAL_IMAGE_BIN is not set` | Set env to `scripts/image/sd_external_image.sh` |
| `OLLAMA_SD_CLI must point to sd-cli` | Run `install_stable_diffusion.sh` |
| Structured noise / stripes | Confirm `--diffusion-fa` (manifest `diffusion_fa: true`) |
| Vulkan OOM | `--vae-on-cpu`, lower to 512×512, `zerollama stop` other models |
| Slow | Expected on A380; SD1.5 @ 512 is minutes-class for 20 steps |

---

## Files

| File | Role |
|------|------|
| `scripts/image/install_stable_diffusion.sh` | Download/build sd-cli + HF weights |
| `scripts/image/sd_external_image.sh` | External-image hook |
| `scripts/image/register_sd_models.sh` | Write config-only manifest |
| `modelfiles/sd15-vulkan/` | SD 1.5 Q4 preset |
| `modelfiles/sd15-q8-vulkan/` | SD 1.5 Q8 preset |
| `modelfiles/sd15-turbo-vulkan/` | SD-Turbo (~4 steps) |
| `modelfiles/sdxl-vulkan/` | SDXL Q4 @ 768 (experimental 6 GB) |
| `server/modality/external_image.go` | Passes `backend_paths` + `image_generation` env |
