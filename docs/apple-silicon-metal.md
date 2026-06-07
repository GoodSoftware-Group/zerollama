# Apple Silicon & Metal — operator guide

**Audience:** macOS users on M-series (and Intel Mac with Metal) running zerollama locally.

**Related:** [development.md](./development.md) (build), [gpu-5080-operator-guide.md](./gpu-5080-operator-guide.md) (CUDA counterpart), [phase13-runtime-vram.md](./phase13-runtime-vram.md), [ROADMAP.md](./ROADMAP.md#apple-silicon--metal-track).

---

## Why this guide exists

Apple Silicon does **not** share the same stack as the RTX 5080 CUDA path:

| Concern | NVIDIA Linux box | Apple Silicon |
|---------|------------------|---------------|
| GPU memory | Discrete VRAM (NVML / `nvidia-smi`) | **Unified memory** (weights + OS on one pool) |
| Default GGUF inference | Runtime + CUDA `llama-server` | **ggml Metal** in the main binary (best tested) |
| Safetensors / MLX-native | N/A or CUDA MLX build | **Optional MLX engine** (Metal) |
| Phase 11/13 probes | `nvidia-smi` autoconfig | **`metal-unified`** via `vm_stat` + `apple_silicon.yaml` |

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
  GGUF text         │  (Metal if llama-server built for it) │
                    └─────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
  safetensors       │  mlxrunner / imagegen (optional MLX)  │  ← --experimental create
  (IsMLX)           │  Not runtime-default                  │
                    └─────────────────────────────────────┘
```

**Recommendation today:** Use **GGUF + default serve** for general chat. Enable **runtime proxy** when you need Phase 12 tools on runtime-routed models. Use **MLX** only for safetensors-native or image MLX models after building the MLX engine.

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

- Build: [development.md](./development.md) — MLX component, Metal toolchain on Apple Silicon.
- `build_darwin.sh` can ship Metal v3/v4 MLX libs.
- Creation: `zerollama create --experimental …`
- **Excluded** from Phase 12 runtime-default routing (`modelEligibleForRuntimeDefault` rejects `IsMLX()`).

**Why keep MLX:** Apple-native weights and image pipelines; **why not merge into runtime yet:** different weight format, subprocess, and long-term may stay a side path (see [python-migration.md](./python-migration.md) Phase 6).

---

## Quick start (Apple Silicon)

### Default GGUF chat

```bash
zerollama serve
zerollama run llama3.2:3b
```

Uses **Metal ggml** — no `LLAMA_SERVER_BIN` required for this path.

### Runtime + tools (embedded sidecar)

```bash
export ZEROLLAMA_RUNTIME_EMBED=1
# Metal-capable llama-server on PATH or:
export LLAMA_SERVER_BIN=/path/to/llama-server
zerollama serve
```

Check `/health` on `:8081`:

- `autoconfig.pick`: `apple_silicon`
- `vram_probe_effective`: `metal-unified` (when checks on)

### Smoke (no NVIDIA required)

```bash
./scripts/phase12_golden_ci.sh          # CI parity (no GPU)
./scripts/macos_metal_smoke.sh          # coordination + /health metal fields
./scripts/gpu_metal_session.sh          # smoke + Phase 13 snapshot (+ optional Phase 14)
# Optional full infer (needs serve + small GGUF + Metal llama-server):
# LLAMA_MODEL=... LLAMA_SERVER_BIN=... RUN_E2E_GPU=1 ./scripts/gpu_metal_session.sh
# Phase 14 inprocess on Metal (needs LLAMA_CPP_LIB):
# RUN_E2E_PHASE14=1 RUN_E2E_INPROCESS=1 LLAMA_CPP_LIB=... ./scripts/gpu_metal_session.sh
```

**Routing policy (GGUF vs runtime vs MLX):** [mlx-routing-policy.md](./mlx-routing-policy.md)

---

## Roadmap (Metal track)

See [ROADMAP.md](./ROADMAP.md#apple-silicon--metal-track). Summary:

| Milestone | Status |
|-----------|--------|
| **M1** Unified probe + `apple_silicon.yaml` autoconfig + host budget on Darwin | **Shipped** |
| **M2** `macos_metal_smoke.sh` + docs + pytest/CI greps | **Shipped** |
| **M3** Phase 14 inprocess on Metal + session autotune | **Started** — `gpu_metal_session.sh` |
| **M4** MLX vs runtime policy doc + routing guards | **Shipped** — [mlx-routing-policy.md](./mlx-routing-policy.md) |

---

## Troubleshooting

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `/health` `vram_probe_effective: skipped` | Checks off or unified fallback off | `ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM=1`, keep unified fallback on |
| `autoconfig.pick: single_gpu` on Mac | Explicit `ZEROLLAMA_RUNTIME_CONFIG` | Unset or point to `apple_silicon.yaml` |
| Runtime 502 host memory on big model | Unified pool too small for MXFP4 mmap | Smaller quant; same as Linux — host RAM not VRAM |
| Tools 501 on Mac | Model not runtime-routed | Manifest backend or `ZEROLLAMA_RUNTIME=1`; ggml path for vision/think |
| MLX model won't load | MLX engine not built | `cmake --install build --component MLX` |

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
