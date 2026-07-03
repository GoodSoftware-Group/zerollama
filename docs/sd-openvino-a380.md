# Stable Diffusion on Intel Arc A380 (OpenVINO GenAI)

**Audience:** Operators running **text-to-image** via **OpenVINO GenAI** on Intel GPU (Arc A380).

**Related:** [sd-vulkan-a380.md](./sd-vulkan-a380.md) (sd.cpp Vulkan path), [a380-runbook.md](./a380-runbook.md), [bench-cache.md](./bench-cache.md)

---

## Why OpenVINO on Arc

| Path | Stack | Best for |
|------|--------|----------|
| `sd15-vulkan` | stable-diffusion.cpp + Vulkan | GGUF weights, same family as llama.cpp; turbo tag for speed |
| **`sd15-openvino`** | OpenVINO GenAI + INT8 IR | Intel GPU-optimized diffusion, pre-exported HF models |

Both use zerollama’s subprocess hook (`openvino-image` or `external-image`). Vulkan and OpenVINO models can coexist — each manifest sets its own wrapper via `backend_paths.external_image_bin`.

**Why per-manifest wrapper:** fleet env `OLLAMA_EXTERNAL_IMAGE_BIN` points at the sd.cpp shell script; OpenVINO needs a Python venv invoke — overriding `external_image_bin` avoids a global env flip when both stacks are installed.

---

## Install

```bash
sudo apt install python3.14-venv   # required once on Ubuntu for the GenAI venv
cd ~/zerollama
chmod +x scripts/install_openvino_diffusion.sh scripts/ov_external_image.sh scripts/register_ov_models.sh

./scripts/install_openvino_diffusion.sh
./scripts/register_ov_models.sh
sudo cp scripts/ov_external_image.sh /usr/lib/ollama-zerollama/
```

Layout:

| Path | Purpose |
|------|---------|
| `/usr/share/zerollama/openvino-genai/venv/` | Python + `openvino` + `openvino-genai` |
| `.../models/sd15-int8-ov/` | `OpenVINO/stable-diffusion-v1-5-int8-ov` |
| `.../models/sdxl-int8-ov/` | `OpenVINO/stable-diffusion-xl-base-1.0-int8-ov` |
| `/usr/lib/ollama-zerollama/ov_external_image.sh` | Wrapper invoked per OV manifest |

**Note:** Vulkan SD models keep `OLLAMA_EXTERNAL_IMAGE_BIN=/usr/lib/ollama-zerollama/sd_external_image.sh`. OpenVINO models override the wrapper in manifest — no global env change needed.

---

## Models

| Tag | Weights | Default |
|-----|---------|---------|
| `sd15-openvino` | SD 1.5 INT8 IR | 512², 20 steps, GPU |
| `sdxl-openvino` | SDXL INT8 IR | 768² (experimental 6 GB) |

---

## Generate

```bash
zerollama run sd15-openvino "a watercolor lighthouse at sunset"
zerollama bench sd15-openvino --epochs 1 --warmup 0 --force
zerollama ls sd15-openvino   # PERF column shows seconds
```

API: same as Vulkan — `POST /api/generate` with `"model": "sd15-openvino"`.

---

## A380 notes

- Device defaults to **`GPU`** (Intel Arc via OpenVINO GPU plugin).
- Unload chat models before SDXL: `zerollama stop gemma4-e2b-qat-xl`.
- Compare against Vulkan: `zerollama bench sd15 --force` then compare **PERF** for `sd15-vulkan` vs `sd15-openvino`.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `openvino_genai not installed` | Re-run `install_openvino_diffusion.sh` |
| `OLLAMA_OV_MODEL_DIR` missing | Register + install models |
| GPU not listed | `source .../venv/bin/activate && python -c "import openvino as ov; print(ov.Core().available_devices)"` — install Intel GPU/OpenCL drivers |
| OOM on SDXL | Use `sd15-openvino` or drop to 512 in API |

---

## Files

| File | Role |
|------|------|
| `scripts/install_openvino_diffusion.sh` | venv + HF model download |
| `scripts/ov_image_generate.py` | OpenVINO GenAI generate |
| `scripts/ov_external_image.sh` | external-image hook |
| `scripts/register_ov_models.sh` | config-only manifests |
| `modelfiles/sd15-openvino/` | SD 1.5 preset |
| `modelfiles/sdxl-openvino/` | SDXL preset |
