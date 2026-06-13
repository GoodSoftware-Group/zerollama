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

### Why MLX is a separate build from ggml

| Stack | Build script | Native libs | Weight format |
|-------|--------------|-------------|---------------|
| **ggml Metal** (default GGUF) | `./scripts/build_zerollama_mac.sh` | In-process CGO + `ggml-metal-embed.metal` | GGUF |
| **MLX** (safetensors) | `./scripts/build_production_mac.sh` | `libmlx.dylib`, `libmlxc.dylib`, `mlx.metallib` | safetensors |

**Why not one script:** MLX pins (`MLX_VERSION`, `MLX_C_VERSION`) track upstream Ollama’s **mlx/mlx-c** repos, not `llama.cpp`. CMake fetches MLX, regenerates Go/C bindings (`go generate`), and compiles a large Metal JIT stack — unrelated to ggml runner CGO. Daily GGUF dev should not pay a 10–15 minute MLX compile.

### Rebuild after pin bump

```bash
# 1. Sibling checkouts at pinned commits (or --clone)
./scripts/ensure_mlx_sources.sh
git -C ../mlx checkout $(cat MLX_VERSION)
git -C ../mlx-c checkout $(cat MLX_C_VERSION)

# 2. Full production (MLX v3/v4 + zerollama binary)
export GOFLAGS=-mod=mod   # WHY: cmake runs go generate during MLX configure
./scripts/build_production_mac.sh

# 3. Verify
./zerollama doctor   # [ok] mlx engine → build/metal-v4/lib/ollama/libmlxc.dylib
```

**Outputs:** `dist/darwin-arm64/lib/ollama/mlx_metal_v3/` (macOS 14+) and `mlx_metal_v4/` (macOS 26+ / Xcode 26 SDK). Dev `./zerollama` from repo root discovers `build/metal-v*/lib/ollama/` via `x/mlxrunner/mlx/dynamic.go`.

- **Daily dev (GGUF only):** `./scripts/build_zerollama_mac.sh` — regenerates ggml Metal embed + qwen35 compat; **does not** rebuild MLX.
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

## Mac smoke gaps fixed (Jun 2026)

Three sign-off failures had different root causes — fixing one often exposed the next:

| Symptom | Why | Fix |
|---------|-----|-----|
| `/v1/chat/completions` stream hangs | Python omitted `[DONE]` on error; Gin buffered SSE until EOF | Runtime always emits `[DONE]`; `copyRuntimeResponseBody` flushes each chunk |
| Legacy ggml + runtime both on Metal | Single device; no OS-level GPU isolation | `darwin_ggml_policy.go` blocks ggml when runtime `llama_server=true`; checks run **before** `PrepareForLegacyRunner` |
| `num_gpu=0` still init Metal | Metal registered at first ggml backend init | `GGML_DISABLE_METAL` before `OnceLoad` on first CPU-only load |

**e2e:** `RUN_E2E_STREAM_MAX` (default 120s); legacy ggml skipped on darwin when runtime holds Metal unless `RUN_E2E_LEGACY_FORCE=1`.

---

## Scheduler errors (HTTP status)

When a request tries to load the **ggml runner** but Darwin policy blocks it, the API returns a structured error (not a generic 500):

| HTTP | Error | Why | What to do |
|------|-------|-----|--------------|
| **400** | `model uses zerollama-runtime inference…` | Manifest or default-on routes this GGUF to the **Python runtime**; ggml load is intentionally skipped | Use `/api/generate` or `/api/chat` without forcing legacy ggml; ensure `ZEROLLAMA_RUNTIME_URL` is set |
| **503** | `darwin: runtime sidecar holds Metal…` | Runtime sidecar has a model on Metal; this model **cannot** use runtime-default routing (vision, MLX, etc.) | Unload runtime model (`POST /internal/training-handoff` on `:8081`), use runtime-routed APIs for text GGUF, or set `ZEROLLAMA_LEGACY_RUNNER=1` to force ggml (dual-Metal risk) |

**Why these exist:** Mac smoke and operators hit silent GPU contention when runtime + legacy ggml both touched Metal. The scheduler now fails fast with actionable messages.

---

## Troubleshooting

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `/health` `vram_probe_effective: skipped` | Checks off or unified fallback off | `ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM=1`, keep unified fallback on |
| `autoconfig.pick: single_gpu` on Mac | Explicit `ZEROLLAMA_RUNTIME_CONFIG` | Unset or point to `apple_silicon.yaml` |
| Runtime 502 host memory on big model | Unified pool too small for MXFP4 mmap | Smaller quant; same as Linux — host RAM not VRAM |
| Tools 501 on Mac | Model not runtime-routed | Manifest backend or `ZEROLLAMA_RUNTIME=1`; ggml path for vision/think |
| MLX model won't load | MLX engine not built or stale dylib after pin bump | `./scripts/ensure_mlx_sources.sh` then `GOFLAGS=-mod=mod ./scripts/build_production_mac.sh`; `zerollama doctor` → mlx engine |
| Embed runtime fails on Mac | System Python 3.9, no torch | Default `zerollama serve` uses uv sidecar; set `ZEROLLAMA_RUNTIME_EMBED=1` only if you know system libpython is 3.10+ |
| `zerollama serve` aborts (Python3.framework) | Stale embed-linked binary | Rebuild; sidecar bootstrap skips embed on Darwin by default |
| Phase 14 inprocess load error | Vision model on pinned llama.cpp | Use text-only GGUF (e.g. Qwen text, not gemma3 vision) |
| `llama_backend_source: env` on Mac | `ZEROLLAMA_RUNTIME_LLAMA_BACKEND` set | Unset env; rely on `apple_silicon.yaml` |
| Inprocess load fails on some GGUF | Vision / pinned llama.cpp mismatch | Auto-fallback to Metal `llama-server` on darwin when `LLAMA_SERVER_BIN` set (`ZEROLLAMA_RUNTIME_INPROCESS_FALLBACK=auto`, default on Mac); check `/health` `llama_backend_fallback`; or use text-only GGUF |
| `/api/train/status` 502 / `No module named 'torch'` | Training venv missing | Restart `zerollama serve` (auto venv) or `./scripts/training_uv_venv.sh --verify` |
| Training job fails with QLoRA error | bitsandbytes is CUDA-only | Use `use_lora: true`, `use_qlora: false` |
| HTTP **400** runtime inference on generate | Model uses runtime-default routing | Do not force ggml; use runtime proxy path (see table above) |
| HTTP **503** darwin Metal contention | Runtime holds Metal; vision/MLX ggml load blocked | Handoff runtime or route via runtime; see table above |
| LM Studio MLX missing from `/api/tags` | MLX safetensors import needs full disk copy (~model size + 512 MiB) | Free space on `OLLAMA_MODELS` volume, or `OLLAMA_LMSTUDIO_LIST_ALL=1` to list anyway (pull still checks space) |
| Qwen 3.5 VL wrong parser/renderer (`clip` family) | VL manifest stored `ModelFamily=clip` from projector layer | Re-create model on current tree, or rely on `PrimaryFamily()` routing (Jun 2026+) |

**Qwen 3.5/3.6 opt-in smoke:** after `./scripts/build_zerollama_mac.sh` and pulling a local tag:

```bash
RUN_E2E_QWEN35_MODEL=qwen3.6:latest ./scripts/qwen35_mac_smoke.sh
# or append to full sign-off:
RUN_E2E_QWEN35=1 RUN_E2E_QWEN35_MODEL=qwen3.6:latest ./scripts/m3_metal_signoff.sh
```

See [qwen35-apple-silicon.md](./qwen35-apple-silicon.md).

---

## Code map

| Piece | Path | Why |
|-------|------|-----|
| Metal scheduler (Go) | `server/sched.go`, `llm/server.go` | Unified-memory layout; darwin contention checks before VRAM handoff |
| Darwin Metal policy | `server/darwin_ggml_policy.go` | Block dual Metal residency (runtime + ggml) on one device |
| VL family routing | `server/model_family.go` | `PrimaryFamily()` picks LLM arch over projector (`clip`) in VL manifests |
| Runtime SSE proxy | `server/runtime_proxy.go` | `copyRuntimeResponseBody` — flush per chunk so SSE does not buffer until EOF |
| MLX runner | `x/mlxrunner/`, `server/sched.go` (`IsMLX`) | Safetensors stay off Python GGUF runtime |
| LM Studio MLX disk | `internal/lmstudio/lmstudio.go`, `server/lmstudio_catalog.go` | MLX import copies full model size; catalog hides unimportable models |
| `num_gpu=0` Metal gate | `ml/backend/ggml/ggml.go`, `ggml-backend-reg.cpp` | CPU-only loads skip Metal registration when first in process |
| Darwin host memory | `runtime/runtime/host_memory.py` | Unified pool admission (not NVML VRAM) |
| Metal-unified probe | `runtime/runtime/gpu_vram.py` | `vm_stat`-based probe for Phase 13 admission |
| Autoconfig | `runtime/runtime/autoconfig.py`, `configs/apple_silicon.yaml` | Daily Mac path: inprocess llama backend |
| Sign-off smokes | `scripts/metal_signoff.sh`, `scripts/qwen35_mac_smoke.sh` | Phase 13–15 gate; opt-in qwen35 legacy ggml |
