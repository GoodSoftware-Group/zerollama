# Apple Silicon & Metal — operator guide

**Audience:** macOS users on M-series (and Intel Mac with Metal) running zerollama locally.

**Related:** [development.md](./development.md) (build), [gpu-5080-operator-guide.md](./gpu-5080-operator-guide.md) (CUDA counterpart), [phase13-runtime-vram.md](./phase13-runtime-vram.md), [ROADMAP.md](./ROADMAP.md#apple-silicon--metal-track).

---

## Serve on Mac

| What | How | Port |
|------|-----|------|
| **Daily use** | `zerollama serve` | Go `:11434` (default); runtime sidecar auto on loopback `:8081` |
| **CI / sign-off** | `./scripts/serve_mac_runtime.sh` | Explicit `:8080` + `:8081` stack for regression scripts |

On Apple Silicon, **`zerollama serve`** ensures `runtime/.venv` (via uv), starts the Python sidecar, enables autoconfig (`ZEROLLAMA_AUTO_CONFIG=1`), and prepares the training venv when `OLLAMA_TRAINING` is on. No wrapper script required.

Run `zerollama doctor` to validate uv venv, Metal `libllama.dylib`, and sidecar `/health`. First-time setup: **`./scripts/mac_setup.sh`** — see [mac-dev-setup.md](./mac-dev-setup.md).

---

## Why this guide exists

Apple Silicon does **not** share the same stack as the RTX 5080 CUDA path:

| Concern | NVIDIA Linux box | Apple Silicon |
|---------|------------------|---------------|
| GPU memory | Discrete VRAM (NVML / `nvidia-smi`) | **Unified memory** (weights + OS on one pool) |
| Default GGUF inference | Runtime + CUDA `llama-server` | **ggml Metal** in the main binary (best tested) |
| Safetensors / MLX-native | N/A or CUDA MLX build | **Optional MLX engine** (Metal) |
| Phase 11/13 probes | `nvidia-smi` autoconfig | **`metal-unified`** via `vm_stat` + `apple_silicon.yaml` |
| LoRA fine-tuning | CUDA + PEFT (`training.py`) | **PyTorch MPS + PEFT** — same PEFT adapter output as CUDA; QLoRA not supported |

Recent zerollama work optimized **CUDA runtime admission** first. This track makes **Metal usable** (correct probes + autoconfig) and documents **what is already optimized** in Go (scheduler) vs **what is still subprocess-default** in Python runtime.

---

## Three inference paths on Mac

```text
                    ┌─────────────────────────────────────┐
  Pulled GGUF       │  Go ggml runner + Metal (default)   │  ← most library models
  (llama3, qwen…)   │  Flash attention, unified mem sched │
                    └─────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
  Runtime-routed    │  Python runtime → llama-server      │  ← Phase 12 tools, manifest gguf
  GGUF text         │  or inprocess (Metal libllama)      │  ← --llama-cpp-backend
                    └─────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
  safetensors       │  mlxrunner / imagegen (optional MLX)  │  ← --experimental create
  (IsMLX)           │  Not runtime-default                  │
                    └─────────────────────────────────────┘
```

**Upstream Ollama (reference):** default GGUF is **Go → llama-server** (no Python sidecar). Zerollama [Phase 17](./ROADMAP.md#phase-17--upstream-gguf-path-alignment-directional) targets that shape for plain text while keeping Python for admission/training. Compare: [upstream-ollama-diff.md](./upstream-ollama-diff.md).

**Recommendation today:** Use **GGUF + default serve** for general chat. Enable **runtime proxy** when you need Phase 12 tools on runtime-routed models, or **`--llama-cpp-backend`** to benchmark against upstream. Use **MLX** only for safetensors-native or image MLX models after building the MLX engine.

**Qwen 3.5 / 3.6 (`qwen35`, `qwen35moe`, e.g. `qwen3.6:latest`):** On Mac, use a **fresh build** (`./scripts/build_zerollama_mac.sh`) and **restart serve**. Routing, compat metadata, and Metal embed requirements are documented in [qwen35-apple-silicon.md](./qwen35-apple-silicon.md)—**why** three separate failure modes appeared in Jun 2026 builds.

---

## What Go already optimizes for Metal

Inherited Ollama scheduler behavior (still true in zerollama):

- **Metal built into** the standard Apple Silicon binary — no extra CUDA steps.
- **No partial-offload penalty** in memory layout (`llm/server.go` treats Metal like full GPU).
- **No discrete VRAM recovery wait** when evicting runners (`sched.go` — unified memory).
- **Phase 8 broker** still applies when runtime + ggml share the process.

---

## Python runtime on Mac (Phase 11/13)

### Autoconfig

When `ZEROLLAMA_AUTO_CONFIG=1` (default) and no `ZEROLLAMA_RUNTIME_CONFIG`:

- **darwin** → `runtime/configs/apple_silicon.yaml` (not `single_gpu.yaml`).
- `/health` → `autoconfig.pick: apple_silicon`, `gpu_total_vram` from `hw.memsize`.

**Why:** `single_gpu.yaml` assumes NVIDIA probes and 16GB discrete headroom; Mac needs unified-memory semantics.

### VRAM probe: `metal-unified`

On macOS without NVIDIA tools, runtime admission uses:

- **`vm_stat`** free + inactive + speculative pages × page size.
- Probe name on `/health`: `vram_probe_effective: metal-unified`.
- Host RAM budget for large GGUF mmap uses the same **`read_host_memory()`** path (fixes silent skip on Darwin).

Disable unified probe: `ZEROLLAMA_RUNTIME_VRAM_UNIFIED_FALLBACK=0`.

### YAML defaults (`apple_silicon.yaml` `vram:`)

| Key | Default | Why lower than CUDA |
|-----|---------|---------------------|
| `min_free` | `512MiB` | OS shares the same pool; aggressive 1GiB floor rejects valid loads. |
| `training_reserve` | `1GiB` | Training + inference still coordinated; less fake “VRAM” to reserve. |
| `clamp_num_ctx` | `"0"` | Same as CUDA — no silent context cut unless you opt in. |

Env always wins over YAML.

---

## MLX engine (optional)

**Not** the default for GGUF. Required for **`ModelFormat: safetensors`** (`IsMLX()`).

- **Daily dev:** `./scripts/build_zerollama_mac.sh` — regenerates `ggml-metal-embed.metal`, links Metal ggml + qwen35 compat; does not rebuild MLX.
- **MLX / release:** `./scripts/build_production_mac.sh` → run from `dist/darwin-arm64/` (see [mac-dev-setup.md](./mac-dev-setup.md)).
- Metal Toolchain (build time): `xcodebuild -downloadComponent MetalToolchain`
- Creation: `zerollama create --experimental …`
- **Excluded** from Phase 12 runtime-default routing (`modelEligibleForRuntimeDefault` rejects `IsMLX()`).

**Startup noise:** `CHECK failed: mlx_distributed_group_new_` usually means a **stale** flat `build/lib/ollama/libmlxc.dylib`. Remove it or use the production layout; `zerollama doctor` warns.

**Why keep MLX:** Apple-native weights and image pipelines; **why not merge into runtime yet:** different weight format, subprocess, and long-term may stay a side path (see [python-migration.md](./python-migration.md) Phase 6).

---

## Quick start (Apple Silicon)

### Default (ggml + runtime sidecar)

```bash
zerollama serve
zerollama run llama3.2:3b
```

Listens on **`OLLAMA_HOST`** (default `:11434`). On Darwin, zerollama also starts the uv runtime sidecar on `:8081` with Metal autoconfig. GGUF chat uses **Metal ggml** in the main binary; runtime tools and admission use the sidecar.

### CI / explicit sidecar stack

```bash
LLAMA_CPP_ROOT=../llama.cpp ./scripts/build_llama_server.sh
export LLAMA_MODEL=/path/to/text-only.gguf   # avoid vision/embed families for inprocess pin
./scripts/serve_mac_runtime.sh
```

Same uv sidecar as `zerollama serve`, but Go listens on `:8080` for smoke scripts.

### LoRA fine-tuning (MPS)

Same **`/api/train/*`** API as CUDA hosts. `zerollama serve` auto-creates `.venv-training` when training is enabled:

```bash
zerollama serve
curl -s http://127.0.0.1:11434/api/train/status | jq .
```

Manual verify: `./scripts/training_uv_venv.sh --verify`. Submit with **`use_lora: true`**, **`use_qlora: false`**. Adapter output under `output_dir/lora_adapter/` is PEFT-compatible — use **`ADAPTER`** in a Modelfile against the same HF base model. Details: [gpu-training.md](./gpu-training.md#apple-silicon-mps).

Check `/health` on `:8081`:

- `autoconfig.pick`: `apple_silicon`
- `llama_backend`: `inprocess`, `llama_backend_source`: `config`
- `vram_probe_effective`: `metal-unified` (when checks on)

**Embed sidecar** (Linux or Mac with Python 3.10+ system libpython only):

```bash
export ZEROLLAMA_RUNTIME_EMBED=1
export LLAMA_SERVER_BIN=/path/to/llama-server
zerollama serve
```

### Smoke (no NVIDIA required)

```bash
./scripts/phase12_golden_ci.sh          # CI parity (no GPU)
./scripts/macos_metal_smoke.sh          # coordination + /health metal fields
./scripts/m3_metal_signoff.sh           # M3 gate: Phase 13 snapshot + Phase 14 inprocess Metal
./scripts/metal_signoff.sh              # Full gate: M3 + Phase 15 (recommended one-shot)
./scripts/phase15_metal_signoff.sh      # Phase 15 KV hook + multi-seq only (sidecar)
./scripts/gpu_metal_session.sh          # macos_metal_smoke + snapshot (+ optional Phase 14/15)
# Self-contained session (starts sidecar + Go):
# METAL_SELF_START=1 RUN_E2E_PHASE14=1 RUN_E2E_PHASE15=1 ./scripts/gpu_metal_session.sh
# Prerequisite for Phase 14 inprocess:
#   LLAMA_CPP_ROOT=../llama.cpp ./scripts/build_llama_server.sh   # builds libllama.dylib + llama-server (Metal)
# Optional full infer (needs serve + small GGUF + Metal llama-server):
# LLAMA_MODEL=... LLAMA_SERVER_BIN=... RUN_E2E_GPU=1 ./scripts/gpu_metal_session.sh
# Phase 14 inprocess on Metal (apple_silicon.yaml default on darwin):
# RUN_E2E_PHASE14=1 LLAMA_MODEL=... ./scripts/gpu_metal_session.sh
```

**Full sign-off (recommended):** `./scripts/metal_signoff.sh` — M3 + Phase 15 in one command.

**M3 only:** `./scripts/m3_metal_signoff.sh` ensures `runtime/.venv` via **uv**, starts sidecar runtime + Go proxy, runs coordination + snapshot + **`phase14_yaml_config_smoke.sh`** (inprocess from `apple_silicon.yaml`). Default model: smallest local **text** GGUF (skips embed/vision models). Override with `M3_LLAMA_MODEL=/path/to/model.gguf`. Add Phase 15: `RUN_E2E_PHASE15=1 ./scripts/m3_metal_signoff.sh`.

**Routing policy (GGUF vs runtime vs MLX):** [mlx-routing-policy.md](./mlx-routing-policy.md)

---

## Compare with upstream Ollama

Clone vanilla Ollama for Metal A/B without merging zerollama:

```bash
./scripts/clone_upstream_ollama.sh
cd ../ollama-upstream && go build -o ollama .
OLLAMA_HOST=127.0.0.1:11435 ./ollama serve
```

| Arm | Port | GGUF stack |
|-----|------|------------|
| Zerollama default | `:11434` | Go → ggml Metal runner |
| Zerollama `--llama-cpp-backend` | `:11434` + sidecar `:8081` | Go → Python → inprocess / llama-server Metal |
| Upstream Ollama | `:11435` | Go → llama-server Metal |

Benchmark all three with `go run ./cmd/bench` (see [llama-cpp-backend.md](./llama-cpp-backend.md)). Roadmap milestone **M7** (done): keep ggml default — [phase17-llama-server.md](./phase17-llama-server.md), [ROADMAP.md](./ROADMAP.md#apple-silicon--metal-track).

---

## Roadmap (Metal track)

See [ROADMAP.md](./ROADMAP.md#apple-silicon--metal-track). Summary:

| Milestone | Status |
|-----------|--------|
| **M1** Unified probe + `apple_silicon.yaml` autoconfig + host budget on Darwin | **Shipped** |
| **M2** `macos_metal_smoke.sh` + docs + pytest/CI greps | **Shipped** |
| **M3** Phase 14 inprocess on Metal + session autotune | **Shipped** — `./scripts/m3_metal_signoff.sh` |
| **M4** MLX vs runtime policy doc + routing guards | **Shipped** — [mlx-routing-policy.md](./mlx-routing-policy.md) |
| **M5** Phase 15 KV + multi-seq on Metal | **Shipped** — `./scripts/metal_signoff.sh` |
| **M6** MPS LoRA training (PEFT adapters) | **Shipped** — same `/api/train` + `lora_adapter/` output as CUDA |
| **M7** Upstream-shape GGUF benchmark | **Done** — ggml ~164 vs upstream ~158 tok/s @ 4k ctx; [phase17-llama-server.md](./phase17-llama-server.md) |

## Troubleshooting

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `/health` `vram_probe_effective: skipped` | Checks off or unified fallback off | `ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM=1`, keep unified fallback on |
| `autoconfig.pick: single_gpu` on Mac | Explicit `ZEROLLAMA_RUNTIME_CONFIG` | Unset or point to `apple_silicon.yaml` |
| Runtime 502 host memory on big model | Unified pool too small for MXFP4 mmap | Smaller quant; same as Linux — host RAM not VRAM |
| Tools 501 on Mac | Model not runtime-routed | Manifest backend or `ZEROLLAMA_RUNTIME=1`; ggml path for vision/think |
| MLX model won't load | MLX engine not built | `cmake --install build --component MLX` |
| Embed runtime fails on Mac | System Python 3.9, no torch | Default `zerollama serve` uses uv sidecar; set `ZEROLLAMA_RUNTIME_EMBED=1` only if you know system libpython is 3.10+ |
| `zerollama serve` aborts (Python3.framework) | Stale embed-linked binary | Rebuild; sidecar bootstrap skips embed on Darwin by default |
| Phase 14 inprocess load error | Vision model on pinned llama.cpp | Use text-only GGUF (e.g. Qwen text, not gemma3 vision) |
| `llama_backend_source: env` on Mac | `ZEROLLAMA_RUNTIME_LLAMA_BACKEND` set | Unset env; rely on `apple_silicon.yaml` |
| Inprocess load fails on some GGUF | Vision / pinned llama.cpp mismatch | Auto-fallback to Metal `llama-server` on darwin when `LLAMA_SERVER_BIN` set (`ZEROLLAMA_RUNTIME_INPROCESS_FALLBACK=auto`, default on Mac); check `/health` `llama_backend_fallback`; or use text-only GGUF |
| `/api/train/status` 502 / `No module named 'torch'` | Training venv missing | Restart `zerollama serve` (auto venv) or `./scripts/training_uv_venv.sh --verify` |
| Training job fails with QLoRA error | bitsandbytes is CUDA-only | Use `use_lora: true`, `use_qlora: false` |

---

## Code map

| Piece | Path |
|-------|------|
| Metal scheduler (Go) | `server/sched.go`, `llm/server.go` |
| MLX runner | `x/mlxrunner/`, `server/sched.go` (`IsMLX`) |
| Darwin host memory | `runtime/runtime/host_memory.py` |
| Metal-unified probe | `runtime/runtime/gpu_vram.py` |
| Autoconfig | `runtime/runtime/autoconfig.py`, `configs/apple_silicon.yaml` |
| Smoke | `scripts/macos_metal_smoke.sh` |
