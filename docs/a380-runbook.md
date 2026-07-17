# Intel Arc A380 runbook — what to run (Jul 2026)

**Status:** Initial operator target for **6 GB GDDR6** Intel Arc A380 on Linux via **Vulkan (Mesa ANV)**.

**Research lane:** [`~/bmtl/asm_lab/lanes/arc-a380`](../../bmtl/asm_lab/lanes/arc-a380) — measured pitfalls, fixtures, and `runs/research_synthesis.json`. Zerollama scripts source thresholds from that work.

**CUDA counterpart:** [5080-runbook.md](./5080-runbook.md) — RTX 5080 uses an entirely different stack.

---

## Start here (three commands)

```bash
cd ~/zerollama
git pull

source ./scripts/gpu/a380_env.sh
./scripts/gpu/a380_signoff.sh --build
```

| Script | Role |
|--------|------|
| [`scripts/gpu/a380_env.sh`](../scripts/gpu/a380_env.sh) | Vulkan env, sign-off model paths, research lane pointer |
| [`scripts/gpu/a380_signoff.sh`](../scripts/gpu/a380_signoff.sh) | Tiered sign-off (device → profile → API smoke) |
| [`scripts/gpu/gpu_a380_session.sh`](../scripts/gpu/gpu_a380_session.sh) | Quick session when serve is already configured |
| [`scripts/serve/serve_a380_example.sh`](../scripts/serve/serve_a380_example.sh) | Production-shaped `zerollama serve` for Vulkan |
| [`scripts/gpu/a380_vulkan_smoke.sh`](../scripts/gpu/a380_vulkan_smoke.sh) | API benchmark with load_ms / total_duration checks |

---

## What works on A380 (and what does not)

| Capability | A380 |
|------------|------|
| **GGUF inference (Vulkan)** | Yes — primary path (`OLLAMA_VULKAN=1`, Mesa ANV) |
| **Partial GPU offload** (`num_gpu` 1–12, `-ngl` hybrid) | **No** — timeouts / cliff; use **full GPU** or CPU-only |
| **GPU training** (`/api/train/*`) | **No** — CUDA / MPS only |
| **MLX engine** | **No** — Metal or NVIDIA CUDA 13+ |
| **Image gen (SD 1.5 Vulkan)** | **Yes** — `sd15-vulkan` via [sd-vulkan-a380.md](./sd-vulkan-a380.md) |
| **Image gen (OpenVINO INT8)** | **Yes** — `sd15-openvino` via [sd-openvino-a380.md](./sd-openvino-a380.md) |
| **SYCL / Level Zero** | Lab control only — **not** zerollama production backend |

---

## Required environment (WHY each knob)

| Variable | Value | Why |
|----------|-------|-----|
| `OLLAMA_VULKAN` | `1` | Enable ggml Vulkan backend discovery |
| `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT` | `1` | Mesa ANV integer-dot path unstable on A380 (asm_lab systemd) |
| `OLLAMA_LLM_LIBRARY` | `vulkan` | Skip CUDA autodetection on Intel hosts |
| `OLLAMA_TRAINING` | `false` | No CUDA training on Arc |
| `ZEROLLAMA_GPU_PROFILE_ID` | `arc-a380` | L1 profile when runtime enabled (`runtime/configs/gpu/arc-a380.json`) |
| `OLLAMA_HOST` | `192.168.255.105:11434` | **Private network bind** — agents hit this LAN IP directly; do **not** SSH port-forward `:11434` |

Build with Vulkan SDK or distro `vulkan-sdk` / `libvulkan1`. User needs `/dev/dri` access (`render` / `video` group).

---

## Sign-off fixture

Research lane standardized on **Tiny-Agent 0.5B Q8_0**:

| Item | Path / tag |
|------|------------|
| GGUF | `/root/models/tiny-agent/Tiny-Agent-a-0.5B.Q8_0.gguf` |
| Ollama tag | `tiny-agent:q8` |
| Import | Copy GGUF to `/var/lib/ollama/imports/`, `zerollama create` from Modelfile |

**Why Q8_0:** f16 + wrong ggml paths are a trap (~0.5 tok/s); Q8_0 matches production quant and fits 6 GB VRAM.

---

## Metrics — what to cite (critical)

From asm_lab `runs/ollama_vulkan.json` and `runs/ollama_load_investigation.json`:

| Metric | Typical (`tiny-agent:q8`) | Use |
|--------|---------------------------|-----|
| `eval_tok_s` | ~43 tok/s | **Misleading alone** — ignores ~580 ms `load_duration` every request |
| `load_ms` | ~582 ms | Paid on **every** `/api/generate`; `keep_alive` does **not** remove it |
| `total_duration_eval_tok_s` @ 8 tok | ~10 tok/s | **Honest short-request UX** |
| `total_duration_eval_tok_s` @ 256 tok | ~38 tok/s | Long completions amortize load tax |

**Do not claim** “GPU matches CPU at ~43 tok/s” without stating decode length and citing `total_duration_eval_tok_s`.

---

## Measured pitfalls (from research lane)

1. **Per-request `load_ms` ~580 ms** — structural API overhead, not cold-start only.
2. **Partial `num_gpu` fails** — only full offload (`99` / all layers) is reliable.
3. **Multi-model residency** — stop unused models (`zerollama stop …`) before benchmarks; dual residency can halve eval tok/s.
4. **ReBAR off** — host had Resizable BAR disabled; SYCL transfer rows are pessimistic (lab only).
5. **Integer dot on ANV** — keep `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1`.

Full narrative: [`~/bmtl/asm_lab/lanes/arc-a380/RESEARCH_NOTES.md`](../../bmtl/asm_lab/lanes/arc-a380/RESEARCH_NOTES.md) §22–24.

---

## L1 GPU profile (`arc-a380`)

File: [`runtime/configs/gpu/arc-a380.json`](../runtime/configs/gpu/arc-a380.json)

| Field | Value | Why |
|-------|-------|-----|
| `ctx_size` | 4096 | 6 GB VRAM headroom |
| `n_parallel` | 1 | Contention + load tax on small VRAM |
| `n_gpu_layers` | 999 (→ all) | Partial offload unsupported |
| `flash_attn` | false | Vulkan path conservative default |
| `batch_size` / `ubatch_size` | 256 / 64 | Below 5080-class defaults |

Force profile: `ZEROLLAMA_GPU_PROFILE_ID=arc-a380`. Vulkan name match via `vulkaninfo` when unset.

Runtime YAML: [`runtime/configs/arc_a380.yaml`](../runtime/configs/arc_a380.yaml).

---

## Production deploy (full stack)

Zerollama is **two binaries**: `zerollama` (Go API/scheduler) and **vendor `llama-server`** (eliza @ `LLAMA_CPP_COMMIT` + Ollama patches). `go build` alone leaves inference on stock `/usr/lib/ollama` — no TBQ/QJL fork KV types.

```bash
cd ~/zerollama
./scripts/build/build_zerollama_a380.sh              # vendor clone + Vulkan llama-server + go build
sudo cp zerollama /usr/bin/zerollama
sudo ./scripts/gpu/install_a380_llama_server.sh    # → /usr/lib/ollama-zerollama + /etc/zerollama/a380-llama.env
sudo cp scripts/zerollama-a380.service /etc/systemd/system/zerollama.service
sudo ln -sf /usr/bin/zerollama /usr/bin/ollama
sudo systemctl daemon-reload && sudo systemctl enable --now zerollama
sudo systemctl disable --now ollama
```

Verify fork inference binary:

```bash
/usr/lib/ollama-zerollama/llama-server --help 2>&1 | grep -E 'tbq3_0|qjl1_256'
source ./scripts/gpu/a380_env.sh && ./scripts/gpu/a380_signoff.sh --no-serve
```

---

## Local image generation

**Why not MLX imagegen:** Z-Image / Flux need ~16 GB CUDA or Metal; A380 has 6 GB GDDR6. **Why two local stacks:** sd.cpp reuses Vulkan/ggml tooling; OpenVINO GenAI offers Intel-tuned INT8 IR — bench both and pick via **PERF**.

```bash
# One-time install (see linked docs for deps)
./scripts/image/install_stable_diffusion.sh
./scripts/image/register_sd_models.sh
./scripts/image/install_openvino_diffusion.sh    # optional
./scripts/image/register_ov_models.sh

# Service env (Vulkan default wrapper)
# OLLAMA_EXTERNAL_IMAGE_BIN=/usr/lib/ollama-zerollama/sd_external_image.sh

zerollama run sd15-turbo-vulkan "a red apple on white"
zerollama run sd15-openvino "a lighthouse at sunset"
zerollama ls image
zerollama bench sd15 --force --epochs 1 --warmup 0
zerollama stop gemma4-e2b-qat-xl    # free VRAM before SDXL
```

| Tag | Stack | Typical PERF (A380) |
|-----|--------|---------------------|
| `sd15-vulkan` | sd.cpp Q4 | minutes-class @ 512², 20 steps |
| `sd15-turbo-vulkan` | sd.cpp SD-Turbo | ~30–40 s @ 512², 4 steps |
| `sd15-openvino` | OpenVINO INT8 GPU | ~50–60 s @ 512² |
| `sdxl-vulkan` / `sdxl-openvino` | SDXL | experimental 768² — unload chat first |

Docs: [sd-vulkan-a380.md](./sd-vulkan-a380.md), [sd-openvino-a380.md](./sd-openvino-a380.md), [bench-cache.md](./bench-cache.md).

**Deps (Vulkan llama-server build):** `cmake`, `golang-go`, `gcc`, `g++`, `vulkan-tools`, `libvulkan-dev`, `glslang-tools`.

---

## Production serve

```bash
# Option A — wrapper (recommended; uses vendor llama-server when installed)
source ./scripts/gpu/a380_env.sh
bash ./scripts/serve/serve_a380_example.sh

# Option B — systemd (loads EnvironmentFile=/etc/zerollama/a380-llama.env)
# See scripts/zerollama-a380.service
```

**Inference-only fast start:**

```bash
OLLAMA_TRAINING=false ZEROLLAMA_RUNTIME_EMBED=off ./zerollama serve
```

---

## Re-run asm_lab probes (optional)

```bash
# Agents: OLLAMA_HOST=http://192.168.255.105:11434 on the private network — no SSH -L port-forward
cd ~/bmtl/asm_lab/lanes/arc-a380
make ollama-vulkan ollama-load-investigation ollama-decode-scaling
make research-synthesis
```

Or from zerollama:

```bash
A380_RUN_RESEARCH=1 ./scripts/gpu/gpu_a380_session.sh
```

---

## BIOS / host follow-ups (research lane)

| Action | Why |
|--------|-----|
| **Enable Resizable BAR** | Re-measure SYCL / transfer-bound rows (lab) |
| `perf_stream_paranoid=0` | EU / XmxActive metrics for mechanism probes |
| Intel GPU driver stack | [Intel client GPU docs](https://dgpu-docs.intel.com/driver/client/overview.html) |

---

## Related docs

- [sd-vulkan-a380.md](./sd-vulkan-a380.md) — stable-diffusion.cpp on Mesa ANV
- [sd-openvino-a380.md](./sd-openvino-a380.md) — OpenVINO GenAI INT8 path
- [bench-cache.md](./bench-cache.md) — PERF column and `zerollama bench sd15`
- [gpu.mdx](./gpu.mdx) — Vulkan experimental flag, Intel driver link
- [development.md](./development.md) — building with Vulkan SDK
- [gpu-profiles-l1.md](./gpu-profiles-l1.md) — L1 autotune (NVIDIA + Apple; Arc via `arc-a380` profile)
