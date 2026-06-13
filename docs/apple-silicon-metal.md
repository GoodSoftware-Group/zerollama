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

Run `zerollama doctor` to validate uv venv, Metal `libllama.dylib`, and sidecar `/health`. First-time setup: **`./scripts/dev_bootstrap.sh`** (or `./scripts/mac_setup.sh`) — see [mac-dev-setup.md](./mac-dev-setup.md).

### Onboarding tiers (why not one `mac_setup` command)

**Why tiers exist:** sign-off and qwen smokes need **pulled models** and CI uses **`:8080`**, but daily serve uses **`:11434`**. Running everything in default `mac_setup` failed fresh clones.

| Tier | You have | Command |
|------|----------|---------|
| **0** | Xcode CLI, Go, uv only | `./scripts/dev_bootstrap.sh` |
| **1** | Tier 0 + any pulled tag | `./zerollama pull llama3.2:3b` |
| **2** | Tier 1 (local text GGUF) | `MAC_SETUP_SIGNOFF=1 MAC_SETUP_GO=0 MAC_SETUP_BUILD=0 ./scripts/mac_setup.sh` |
| **3** | Tier 1 + qwen tag | `RUN_E2E_QWEN35_MODEL=your:tag ./scripts/qwen35_mac_smoke.sh` |

**Sibling repos (auto on tier 0):** `../llama.cpp` at pin `LLAMA_CPP_VERSION` — runtime inprocess needs `libllama.dylib`. Optional: `../mlx` for safetensors only.

ROADMAP: [M14](./ROADMAP.md#apple-silicon--metal-track).

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

### GPU profile autotune (L1)

**Why separate from Phase 13:** Phase 13 estimates whether a GGUF fits and suggests max `num_ctx`. L1 picks **throughput** flags — batch size, micro-batch, parallel slots, flash-attn, MTP draft bounds — tuned per hardware class. Without L1, every Mac used the same conservative YAML defaults while a 128 GiB M4 Max left parallel slots and batch headroom unused.

**Why RAM tiers, not “Apple Silicon” as one bucket:** unified memory size varies within the same chip generation (e.g. M4 Max 64 GiB vs 128 GiB). `hw.memsize` is the stable selector; chip marketing names are not.

When `ZEROLLAMA_GPU_PROFILE` is not off (default on), loading `apple_silicon.yaml` merges tuned llama-server flags from `runtime/configs/gpu/apple_silicon_*.json`:

| Unified RAM | Profile id | Typical `n_parallel` | Default `-c` | Why this tier |
|-------------|------------|----------------------|--------------|---------------|
| ≤16 GiB | `apple-silicon-16g` | 1 | 16384 | OS + apps share the pool; single slot limits KV duplication |
| ≤24 GiB | `apple-silicon-24g` | 1 | 32768 | Modest headroom; avoid parallel KV on tight hosts |
| ≤48 GiB | `apple-silicon-48g` | 2 | 32768 | Pro/Max class baseline (original L1 conservative target) |
| ≤96 GiB | `apple-silicon-96g` | 4 | 65536 | High-memory Pro/Ultra; long ctx without 128g parallel |
| >96 GiB | `apple-silicon-128g` | 8 | 131072 | Fork: `tbq4_0` K / `tbq3_0` V; PA pool **8192×16** tokens |

**Observability:** `/health` → `gpu_profile` (`id`, `bucket_label`, `unified_memory_gb`, `n_parallel`, `emit_ctx_size`).

**Operator overrides:**

| Control | Why |
|---------|-----|
| `ZEROLLAMA_GPU_PROFILE=0` | A/B vs YAML-only or broken detection |
| `ZEROLLAMA_GPU_PROFILE_CTX=0` | Profile `-c` caps global server ctx; skip when models need 128k+ training ctx |
| `LLAMA_SERVER_EXTRA_ARGS=-c …` | Appended after profile flags; wins on duplicate `-c` |

**Sign-off:** `./scripts/metal_signoff.sh` (Phase 13 snapshot + inprocess Metal + optional Phase 15). `macos_metal_smoke.sh` prints `gpu_profile` from `/health`.

**L2 fork A/B (runtime subprocess path):** `./scripts/l2_metal_bench.sh` — restarts sidecar with stock vs eliza `llama-server`, writes `/tmp/l2-metal-bench.json`. Requires `M3_LLAMA_MODEL` or a local text GGUF. See [gpu-profiles-l2.md](./gpu-profiles-l2.md).

Full reference (NVIDIA buckets, fork safety, file layout): [gpu-profiles-l1.md](./gpu-profiles-l1.md).

Env always wins over YAML for host/port/KV blocks; profile flags merge at config load.

---

## MLX engine (optional)

**Not** the default for GGUF. Required for **`ModelFormat: safetensors`** (`IsMLX()`).

### Why MLX uses a shared dylib script (not duplicate cmake in every entrypoint)

| Stack | Build script | Native libs | Weight format |
|-------|--------------|-------------|---------------|
| **ggml Metal** (default GGUF) | `./scripts/build_zerollama_mac.sh` | In-process CGO + `ggml-metal-embed.metal` | GGUF |
| **MLX** (safetensors) | Same script with `BUILD_MLX=auto` (default when `../mlx` exists), or `./scripts/build_mlx_dylibs_mac.sh` | `libmlx.dylib`, `libmlxc.dylib`, `mlx.metallib` under `build/metal-v*/` | safetensors |

**Why not always compile MLX:** MLX pins (`MLX_VERSION`, `MLX_C_VERSION`) track **mlx/mlx-c**, not `llama.cpp`. CMake + `go generate` for MLX is a 10–15 minute compile unrelated to ggml CGO. **`BUILD_MLX=auto`** skips when dylibs already exist; **`BUILD_MLX=0`** skips entirely for fast GGUF-only iteration.

### Rebuild after pin bump

```bash
# 1. Sibling checkouts at pinned commits (or --clone)
./scripts/ensure_mlx_sources.sh
git -C ../mlx checkout $(cat MLX_VERSION)
git -C ../mlx-c checkout $(cat MLX_C_VERSION)

# 2. Dev (repo-root ./zerollama + build/metal-v*/ dylibs)
BUILD_MLX=1 ./scripts/build_zerollama_mac.sh

# 2b. Release layout (dist/darwin-arm64/)
./scripts/build_production_mac.sh

# 3. Verify
./zerollama doctor   # [ok] mlx engine → build/metal-v*/lib/ollama/...
```

**Outputs:** dev `build/metal-v3/lib/ollama/mlx_metal_v3/` (macOS 14+) and `build/metal-v4/.../mlx_metal_v4/` (macOS 26+); production mirrors under `dist/darwin-arm64/lib/ollama/`. Repo-root `./zerollama` discovers `build/metal-v*/lib/ollama/` via `x/mlxrunner/mlx/dynamic.go`.

- **Daily dev (GGUF only, fast):** `BUILD_MLX=0 ./scripts/build_zerollama_mac.sh`
- **Daily dev (GGUF + safetensors):** `./scripts/build_zerollama_mac.sh` (MLX auto when `../mlx` present)
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

**Full sign-off (recommended):** `./scripts/metal_signoff.sh` — M3 + Phase 15 in one command (starts `:8080` Go + `:8081` runtime sidecar).

**With qwen35 (M4 Max, Jun 2026 PASS):**

```bash
LLAMA_CPP_ROOT=../llama.cpp ./scripts/build_llama_server.sh
RUN_E2E_QWEN35=1 RUN_E2E_QWEN35_MODEL=qwen3.6:latest ./scripts/metal_signoff.sh
```

**Why qwen35 runs before Phase 15 inside the script:** Phase 15’s cleanup stops the runtime sidecar; qwen35 needs `:8081` alive for training-handoff and `runtime_resume_if_needed` after ggml unload.

**M3 only:** `./scripts/m3_metal_signoff.sh` ensures `runtime/.venv` via **uv**, starts sidecar runtime + Go proxy, runs coordination + snapshot + **`phase14_yaml_config_smoke.sh`** (inprocess from `apple_silicon.yaml`). Default model: smallest local **text** GGUF (skips embed/vision models). Override with `M3_LLAMA_MODEL=/path/to/model.gguf`. Add Phase 15: `RUN_E2E_PHASE15=1 ./scripts/m3_metal_signoff.sh`. Add qwen35: `RUN_E2E_QWEN35=1 RUN_E2E_QWEN35_MODEL=…` (same order: qwen35 before Phase 15 when both set).

**Phase 15 multiseq note:** the multiseq step uses a temp YAML with `llama_parallel_slots: 2` and **`ZEROLLAMA_GPU_PROFILE=0`**. **Why:** L1 `apple-silicon-128g` would set `n_parallel=8` and break `kv_inprocess_n_seq_max=2` assertions.

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

## GPU bootstrap discovery (Jun 2026)

**Why this section exists:** M4 Max operators reported `total_vram="0 B"` at serve startup while model-load logs still showed `Apple M4 Max`. That was not missing hardware — it was **broken discovery** in the bootstrap subprocess.

### What happens at startup

After routes are registered, `discover.GPUDevices` spawns a short-lived **ollama-engine** runner and calls `GET /info`. The returned `[]DeviceInfo` drives:

- **`inference compute`** log lines (Metal vs CPU fallback)
- **`vram-based default context`** (`4096` / `32768` / `262144` tiers in `server/routes.go`)
- **GPU layer layout** for ggml and llamarunner loads (`GPULayers` in the load request)

If discovery returns **no GPUs**, the server logs CPU-only compute, `total_vram="0 B"`, and schedules **0 layers to GPU** — even though a separate inference runner subprocess may still initialize Metal later.

### Root cause (fixed Jun 2026)

The ollama-engine `/info` handler used to load a dummy model with **zero GPU layers** to wire up ggml. First init calls `ensureDevices(true)`, which sets `GGML_DISABLE_METAL=1` before `OnceLoad` (`sync.Once` — **first call wins**). Metal never registered; `/info` returned an empty GPU list.

That gate exists **on purpose** for real `num_gpu=0` loads so CPU-only work does not contend with the Python runtime sidecar on one Metal device. Bootstrap discovery must **not** trigger it.

### Fix

| Piece | Path | Why |
|-------|------|-----|
| Discovery probe | `ml/backend/ggml/ggml.go` → `DiscoverBackendDevices()` | `ensureDevices(false)` — enable Metal for enumeration only |
| Bootstrap `/info` | `runner/ollamarunner/runner.go` | No dummy zero-layer model; call `DiscoverBackendDevices()` when unloaded |
| Bootstrap spawn | `discover/runner.go` → `bootstrapDevices()` | Unchanged — still uses ollama-engine runner |

**Verify:**

```bash
BUILD_MLX=0 ./scripts/build_zerollama_mac.sh
./zerollama serve
# expect: inference compute library=Metal … total ~100+ GiB
# expect on first load: offloaded N/N layers to GPU (not 0/N)
```

Quick probe without full serve:

```bash
./zerollama runner --ollama-engine --port 65432 &
curl -s http://127.0.0.1:65432/info | python3 -m json.tool
kill %1
```

### Serve startup time (not the same as discovery)

Discovery is sub-second once fixed. Long `./zerollama serve` gaps are usually **other** work:

| Phase | Typical cost | Why | Faster path |
|-------|--------------|-----|-------------|
| Python runtime sidecar | ~10s | uv venv + `:8081` health | Needed for runtime proxy / tools |
| Blob / manifest prune | 0–60s+ | Scans `OLLAMA_MODELS` | `OLLAMA_NOPRUNE=1` if you accept stale blobs |
| Training worker embed | ~20–30s | Embedded CPython + torch import | `OLLAMA_TRAINING=false` for inference-only |
| First model chat | ~15–40s | Cold load of multi‑GiB GGUF | Warm with `ollama run` or long `OLLAMA_KEEP_ALIVE` |

**Inference-only fast start:**

```bash
OLLAMA_TRAINING=false OLLAMA_NOPRUNE=1 ./zerollama serve
```

First request after listen still pays model load unless the model is already warm.

---

## Go ollama-engine sched_reserve (Jun 2026)

**Why this section:** qwen35moe load on Metal used to abort in `ggml_backend_sched_reserve` with `GGML_ASSERT(tensor->buffer == NULL)`. That looked like a qwen35-specific Metal shader or compat bug; the root cause was **double buffer assignment** in the Go ggml backend.

**Mechanism:** `LoadOperationFit` builds a worst-case graph and calls `sched_reserve` so layer layout knows peak VRAM. The scheduler assigns compute buffers via `ggml_backend_tensor_alloc`. Graph **intermediate** tensors must still have `buffer == NULL` when reserve runs. Eager allocation in `newTensor` (all tensors when `allocMemory=true`) pre-assigned those buffers → assert on large qwen35moe graphs.

**Fix:**

| Piece | Path | Why |
|-------|------|-----|
| Defer graph tensor alloc | `ml/backend/ggml/ggml.go` → `newTensor` | Scheduler owns scratch; only inputs + persistent contexts allocate early |
| `Persistent()` | `ml/backend.go`, kvcache `*.go` | KV/recurrent cells are not graph scratch — they must exist before forward |
| No qwen35 reserve skip | `runner/ollamarunner/runner.go` → `reserveWorstCaseGraph` | Arch blocklist masked the assert; removed after root fix |
| Darwin routing | `llm/server.go` → `pickOllamaEngine` | qwen35\* uses Go engine when `OllamaEngineRequired()` — no darwin legacy gate |

**Verify:** load `qwen3.6:latest` → runner `--ollama-engine`, `offloaded N/N layers to GPU`, generate 200. Opt-in: `./scripts/qwen35_mac_smoke.sh`.

See also [qwen35-apple-silicon.md — sched_reserve](./qwen35-apple-silicon.md#go-engine-sched_reserve-fix-jun-2026).

---

## Metal unified free memory refresh (Jun 2026)

**Why:** `discover.GPUDevices` refreshes free VRAM per device for scheduler admission. On Apple Silicon, Metal “VRAM” is unified host memory. The bootstrap ollama-engine subprocess reports **process-local** free bytes; after loads, scheduler layer fit could see stale or zero free memory on Metal devices.

**Fix:**

| Piece | Path | Why |
|-------|------|-----|
| Refresh on darwin | `discover/runner.go` | Removed early return that skipped free-memory refresh on `darwin/arm64` |
| Host pool fallback | `discover/metal_unified.go` | When Metal device free is unknown, use `GetCPUMem()` (`vm_stat`) capped by device total |
| Cap helper | `discover/metal_unified_cap.go` | Unified free cannot exceed reported device total |

---

## Context length: manifest, `/api/ps`, and load-time KV (Jun 2026)

**Why `/api/ps` disagrees with `/api/show`:** `context_length` in `/api/ps` comes from the **loaded ggml runner** (`llama.ContextLength()`), not the manifest. Updating `parameters.num_ctx` via `/api/create` changes `/api/show` immediately but leaves a warm runner on the old KV size until evicted.

| What you set | Where it shows | Effect |
|--------------|----------------|--------|
| `parameters.num_ctx` in create | `/api/show`, Modelfile | Default merged at **load** — KV pre-sized for this value |
| `options.num_ctx` on chat/generate | Request only (may reload runner if ≠ loaded) | Per-request context; use for Hermes / long prompts |
| Loaded runner | `/api/ps` → `context_length` | Ground truth for **current** in-memory KV |

**Why large manifest defaults hang:** On llamarunner (qwen35 on Mac), `num_ctx` at load reserves KV + recurrent state for the full window. **262144** in the manifest is not the same as passing 262144 once per request — pre-allocation at load can block indefinitely on unified memory.

**Operator pattern:** manifest default **4096** (or 8192); Hermes / clients pass **`options.num_ctx`** when auto-detecting long context. After create or stop, confirm unload with empty `/api/ps` before debugging load.

See [qwen35-apple-silicon.md — manifest vs request num_ctx](./qwen35-apple-silicon.md#manifest-num_ctx-vs-request-optionsnum_ctx-jun-2026).

---

## Mac smoke gaps fixed (Jun 2026)

Three sign-off failures had different root causes — fixing one often exposed the next:

| Symptom | Why | Fix |
|---------|-----|-----|
| `/v1/chat/completions` stream hangs | Python omitted `[DONE]` on error; Gin buffered SSE until EOF | Runtime always emits `[DONE]`; `copyRuntimeResponseBody` flushes each chunk |
| Legacy ggml + runtime both on Metal | Single device; no OS-level GPU isolation | `darwin_ggml_policy.go` blocks ggml when runtime `llama_server=true`; checks run **before** `PrepareForLegacyRunner` |
| `num_gpu=0` still init Metal | Metal registered at first ggml backend init | `GGML_DISABLE_METAL` before `OnceLoad` on first CPU-only load |
| `total_vram="0 B"`, CPU-only offload | Bootstrap `/info` used zero-layer dummy load → `GGML_DISABLE_METAL` in discovery subprocess | `DiscoverBackendDevices()` — see [GPU bootstrap discovery](#gpu-bootstrap-discovery-jun-2026) |
| qwen35moe SIGABRT on Go engine load | `newTensor` eager alloc + `sched_reserve` double-assign | Defer graph alloc; `Persistent()` for KV — see [sched_reserve](#go-ollama-engine-sched_reserve-jun-2026) |
| Full gate fails after qwen35 generate | Phase 15 killed `:8081` before qwen35 resume | qwen35 before Phase 15 in `m3_metal_signoff.sh` |
| Phase 15 multiseq expects `n_seq_max=2`, got 8 | L1 GPU profile `n_parallel=8` on 128 GiB | `ZEROLLAMA_GPU_PROFILE=0` for multiseq temp YAML |
| Phase 14 HTTP 500 `cache_prompt` | L3 bridge passes kwarg; inprocess worker lacked it | Accept/ignore on inprocess + wheel workers |
| Stale Metal free memory on scheduler | Bootstrap subprocess local free bytes | `applyMetalUnifiedFreeMemory` — see [Metal unified free memory](#metal-unified-free-memory-refresh-jun-2026) |

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
| MLX model won't load | MLX engine not built or stale dylib after pin bump | `BUILD_MLX=1 ./scripts/build_zerollama_mac.sh` (or `./scripts/build_production_mac.sh` for dist/); `./scripts/ensure_mlx_sources.sh` at pin bumps; `zerollama doctor` → mlx engine |
| Embed runtime fails on Mac | System Python 3.9, no torch | Default `zerollama serve` uses uv sidecar; set `ZEROLLAMA_RUNTIME_EMBED=1` only if you know system libpython is 3.10+ |
| `zerollama serve` aborts (Python3.framework) | Stale embed-linked binary | Rebuild; sidecar bootstrap skips embed on Darwin by default |
| `/api/show` num_ctx ≠ `/api/ps` context_length | Warm runner not reloaded after create | Rebuild (create evicts runner); or `keep_alive:0` unload — [manifest vs request num_ctx](./qwen35-apple-silicon.md#manifest-num_ctx-vs-request-optionsnum_ctx-jun-2026) |
| Generation hangs after create with huge `num_ctx` | Load-time KV pre-allocation | Revert manifest default to 4096; use `options.num_ctx` per request |
| Phase 14 inprocess load error | Vision model on pinned llama.cpp | Use text-only GGUF (e.g. Qwen text, not gemma3 vision) |
| `llama_backend_source: env` on Mac | `ZEROLLAMA_RUNTIME_LLAMA_BACKEND` set | Unset env; rely on `apple_silicon.yaml` |
| Inprocess load fails on some GGUF | Vision / pinned llama.cpp mismatch | Auto-fallback to Metal `llama-server` on darwin when `LLAMA_SERVER_BIN` set (`ZEROLLAMA_RUNTIME_INPROCESS_FALLBACK=auto`, default on Mac); check `/health` `llama_backend_fallback`; or use text-only GGUF |
| `/api/train/status` 502 / `No module named 'torch'` | Training venv missing | Restart `zerollama serve` (auto venv) or `./scripts/training_uv_venv.sh --verify` |
| Training job fails with QLoRA error | bitsandbytes is CUDA-only | Use `use_lora: true`, `use_qlora: false` |
| HTTP **400** runtime inference on generate | Model uses runtime-default routing | Do not force ggml; use runtime proxy path (see table above) |
| HTTP **503** darwin Metal contention | Runtime holds Metal; vision/MLX ggml load blocked | Handoff runtime or route via runtime; see table above |
| `total_vram="0 B"`, `library=cpu`, `offloaded 0/N layers` | Bootstrap `/info` disabled Metal via zero-layer dummy load (fixed Jun 2026) | Rebuild current tree; see [GPU bootstrap discovery](#gpu-bootstrap-discovery-jun-2026) |
| Slow `./zerollama serve` before listen | Training embed, sidecar, blob prune — not GPU discovery | `OLLAMA_TRAINING=false`, `OLLAMA_NOPRUNE=1` for inference-only |
| `control-looking token … was not control-type` on qwen35 load | Ollama GGUF marks FIM/special tokens as NORMAL; llama.cpp overrides to CONTROL | Harmless — [qwen35-apple-silicon.md — Token warnings](./qwen35-apple-silicon.md#token-warnings-jun-2026) |
| `embeddings required but some input tokens were not marked as outputs` | llamarunner uses `embeddings=true` context for `/api/embed`; chat prefill marks only last token | Harmless — llama.cpp overrides; same doc section |
| LM Studio MLX missing from `/api/tags` | MLX safetensors import needs full disk copy (~model size + 512 MiB) | Free space on `OLLAMA_MODELS` volume, or `OLLAMA_LMSTUDIO_LIST_ALL=1` to list anyway (pull still checks space) |
| Qwen 3.5 VL wrong parser/renderer (`clip` family) | VL manifest stored `ModelFamily=clip` from projector layer | Re-create model on current tree, or rely on `PrimaryFamily()` routing (Jun 2026+) |

**Qwen 3.5/3.6 opt-in smoke:** after `./scripts/build_zerollama_mac.sh` and pulling a local tag:

```bash
RUN_E2E_QWEN35_MODEL=qwen3.6:latest ./scripts/qwen35_mac_smoke.sh
# Full gate (Phase 13–15 + qwen35 — qwen35 runs before Phase 15 in script):
RUN_E2E_QWEN35=1 RUN_E2E_QWEN35_MODEL=qwen3.6:latest ./scripts/metal_signoff.sh
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
| GPU bootstrap discovery | `DiscoverBackendDevices()`, `runner/ollamarunner/runner.go` `/info`, `discover/runner.go` | Bootstrap must call `ensureDevices(false)` — zero-layer dummy load permanently disabled Metal |
| sched_reserve / Persistent | `ml/backend/ggml/ggml.go`, kvcache, `runner/ollamarunner/runner.go` | Graph scratch defers to scheduler; KV buffers allocate eagerly — fixes qwen35moe Metal abort |
| Metal unified free memory | `discover/metal_unified.go`, `discover/runner.go` | Fill Metal free bytes from host pool when bootstrap subprocess values are stale |
| Runtime health cache | `server/inference_workload.go` | 500ms TTL on `/health` probe — training submit idle-wait was per-load RTT |
| Darwin host memory | `runtime/runtime/host_memory.py` | Unified pool admission (not NVML VRAM) |
| Metal-unified probe | `runtime/runtime/gpu_vram.py` | `vm_stat`-based probe for Phase 13 admission |
| Autoconfig | `runtime/runtime/autoconfig.py`, `configs/apple_silicon.yaml` | Daily Mac path: inprocess llama backend |
| Sign-off smokes | `scripts/metal_signoff.sh`, `scripts/qwen35_mac_smoke.sh` | Phase 13–15 gate; opt-in qwen35 Go engine + generate |
