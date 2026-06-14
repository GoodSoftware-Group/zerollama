# L2 — elizaOS/llama.cpp fork evaluation

**Audience:** Contributors evaluating TurboQuant / QJL / PolarQuant vs stock `ggml-org/llama.cpp` (zerollama pin **b9611**).

**Related:** [gpu-profiles-l1.md](./gpu-profiles-l1.md), [ROADMAP — L2](./ROADMAP.md#local-voice--llama-borrowings-eliza-v3), [phase17-llama-server.md](./phase17-llama-server.md), eliza reference `~/Sites/eliza-v3/plugins/plugin-local-inference/native/configs/gpu/SPECS.md`.

---

## Why L2 exists

L1 ships **tuned flags on stock cache types** (`q8_0`). Eliza-v3’s largest wins — **QJL K-cache**, **PolarQuant V-cache**, **TurboQuant**, fused MTP — live in **`elizaOS/llama.cpp`**, not upstream.

**Why not replace vendor immediately:** zerollama’s in-process ggml carries **14 Ollama patches** on `vendor/llama-cpp-b9611/` (GPU discovery, no-alloc, compat overlay, mtmd C API). Blind vendor swap risks qwen35 compat, Metal sign-off, and Phase 15 ctypes layout. L2 is a **gated spike**: measure on 5080 + M-series before merge.

---

## Fork pin (evaluation)

| Field | Value |
|-------|--------|
| Repo | `https://github.com/elizaOS/llama.cpp.git` |
| Ref (Jun 2026) | `96dd1a8466c84bdd419faf3866425260623fb6b0` |
| Sibling tree | `../eliza-llama.cpp` (default) |

---

## Custom KV cache types (fork CLI)

| `--cache-type-k` / `-v` | Role |
|-------------------------|------|
| `qjl1_256` | 1-bit JL K-cache (256-dim sketch) |
| `q4_polar` | 4-bit PolarQuant V-cache |
| `tbq3_0` | TurboQuant 3-bit (often V) |
| `tbq4_0` | TurboQuant 4-bit (often K) |
| `tbq3_tcq` | TurboQuant TCQ K-cache (long ctx) |

L1 profile aliases `turbo3_0` → `tbq3_0`, etc.

**Recommended pairings (from eliza SPECS):**

| GPU class | K | V |
|-----------|---|---|
| RTX 3090 (Ampere) | `q8_0` | `q4_polar` |
| RTX 4090/5080/5090/H200 | `qjl1_256` | `q4_polar` |

Stored in `runtime/configs/gpu/*.json` under `_eliza_fork_llama_server_flags` (active only when fork is enabled).

---

## Fork-only server flags

| Flag | Purpose |
|------|---------|
| `--ctx-checkpoints N` | Voice optimistic-rollback snapshots (eliza duplex); optional for text-only zerollama |
| `--ctx-checkpoint-interval N` | Token interval between checkpoints |

**Detection:** fork `llama-server --help` includes `ctx-checkpoints`. Stock b9611 does not.

---

## Build & run (runtime path)

### macOS / Metal

```bash
# 1. Build fork sibling (Metal)
./scripts/build_eliza_llama_server.sh

# 2. Point runtime at fork binary + enable fork profile merge
export LLAMA_CPP_ROOT=$PWD/../eliza-llama.cpp
export LLAMA_SERVER_BIN=$LLAMA_CPP_ROOT/build/bin/llama-server
export LLAMA_CPP_LIB=$LLAMA_CPP_ROOT/build/bin/libllama.dylib
export ZEROLLAMA_LLAMA_FORK=1   # or omit: auto-probes --help when binary set

# 3. Smoke probe + profile argv
./scripts/l2_fork_eval.sh

# 4. Metal A/B benchmark (stock vs fork decode tok/s + VRAM estimate)
M3_LLAMA_MODEL=/path/to/model.gguf ./scripts/l2_metal_bench.sh
# Output: L2_METAL_BENCH_OUT=/tmp/l2-metal-bench.json (default)

# 5. Runtime subprocess compat (load + generate both binaries)
./scripts/l2_runtime_compat_smoke.sh

# 6. Full gate (eval + compat + bench + verdict)
L2_RUN_27K=1 L2_RUN_131K_FORK=1 ./scripts/l2_full_gate.sh
# 131k leg: fork-only, L2_HIGH_CTX_WARMUPS=2 decode warmups before timed runs

# 7. Serve / sign-off
./scripts/m3_metal_signoff.sh
```

### Linux / CUDA (RTX 5080-class)

```bash
# 1. Build fork sibling (CUDA)
./scripts/build_eliza_llama_server.sh

# 2. Point runtime at fork binary
export LLAMA_CPP_ROOT=$PWD/../eliza-llama.cpp
export LLAMA_SERVER_BIN=$LLAMA_CPP_ROOT/build/bin/llama-server
export LLAMA_CPP_LIB=$LLAMA_CPP_ROOT/build/bin/libllama.so    # .so on Linux
export ZEROLLAMA_LLAMA_FORK=1

# 3. Smoke probe + profile argv
./scripts/l2_fork_eval.sh

# 4. CUDA A/B benchmark (stock vs fork decode tok/s + VRAM)
CUDA_LLAMA_MODEL=/path/to/model.gguf ./scripts/l2_cuda_bench.sh
# Output: L2_CUDA_BENCH_OUT=/tmp/l2-cuda-bench.json (default)

# 5. Runtime compat smoke (Linux variant — uses linux_runtime_serve_lib + .so)
CUDA_LLAMA_MODEL=/path/to/model.gguf ./scripts/l2_cuda_runtime_compat_smoke.sh

# 6. Full CUDA gate (eval + compat + bench legs + verdict)
L2_RUN_27K=1 L2_RUN_131K_FORK=1 ./scripts/l2_cuda_full_gate.sh
# 131k leg: fork-only, L2_HIGH_CTX_WARMUPS=2 decode warmups before timed runs

# 7. Serve / sign-off
./scripts/gpu_5080_session.sh
```

**Env:**

| Variable | Effect |
|----------|--------|
| `ZEROLLAMA_LLAMA_FORK=1` | Force fork profile merge (QJL/Polar, checkpoints) |
| `ZEROLLAMA_LLAMA_FORK=0` | Force stock sanitize (L1 default) |
| *(unset)* | Auto: probe `LLAMA_SERVER_BIN --help` for `--ctx-checkpoints` |
| `ELIZA_LLAMA_CPP_ROOT` / `ELIZA_LLAMA_CPP_REF` | Override clone path / commit for build script |

---

## Runtime integration (shipped)

| Component | Role |
|-----------|------|
| `runtime/llama_fork.py` | Fork detection (env + binary probe) |
| `runtime/gpu_profiles.py` | `_eliza_fork_llama_server_flags` merge; emit `--ctx-checkpoints` when present |
| `/health` | `llama_fork` object + `gpu_profile.llama_fork` |
| `scripts/build_eliza_llama_server.sh` | Isolated fork build |
| `scripts/l2_fork_eval.sh` | Probe + pytest smoke |
| `scripts/l2_metal_bench.sh` | Darwin A/B: stock vs fork decode tok/s + VRAM JSON |
| `scripts/l2_cuda_bench.sh` | Linux/CUDA A/B: stock vs fork decode tok/s + VRAM JSON |
| `scripts/l2_runtime_compat_smoke.sh` | Darwin subprocess compat: load+generate on stock vs fork |
| `scripts/l2_cuda_runtime_compat_smoke.sh` | Linux subprocess compat: mirrors compat smoke with `.so` + `linux_runtime_serve_lib` |
| `scripts/l2_gate_report.sh` | Verdict from one or more bench JSON files |
| `scripts/l2_full_gate.sh` | Darwin gate orchestrator: eval + compat + bench legs + report |
| `scripts/l2_cuda_full_gate.sh` | CUDA gate orchestrator: same structure as Metal gate |
| `scripts/linux_runtime_serve_lib.sh` | Shared sidecar start/stop helpers for Linux (mirrors `macos_runtime_serve_lib.sh`) |

**WHY sibling tree first:** `vendor/llama-cpp-b9611/` + `llama/patches/` stay on b9611 until the gate passes.

---

## M-series sign-off (Jun 2026, M4 Max 128 GiB)

| Model | ctx | Stock | Fork | Notes |
|-------|-----|-------|------|-------|
| eliza-1-2b | 8192 | **37.6 tok/s**, q8_0 | 20.5 tok/s, tbq4_0/tbq3_0 | Stock wins decode + VRAM est |
| eliza-1-27b | 26624 | 13.2 tok/s, q8_0 | 12.7 tok/s, tbq | Stock wins decode (~4%); VRAM est heuristic favors stock (TBQ not modeled) |
| eliza-1-27b | 131072 | Rejected (KV est) | **5.0 tok/s** | Fork-only; `ZEROLLAMA_GPU_PROFILE_CTX=0` + `runtime_kv` 8192 blocks |

JSON: `/tmp/l2-metal-bench.json`, `/tmp/l2-gate/`. **Runtime compat smoke:** PASS (stock + fork subprocess generate).

**Gate status (Metal):** **FAIL merge** at small ctx (stock faster). Fork may still win **max ctx + VRAM** at 27k+ — run `L2_RUN_27K=1 L2_RUN_131K_FORK=1 ./scripts/l2_full_gate.sh`. **Gate status (CUDA 5080):** **Not run** — scripts ready (`l2_cuda_bench.sh`, `l2_cuda_full_gate.sh`); run `CUDA_LLAMA_MODEL=/path/to/model.gguf L2_RUN_27K=1 L2_RUN_131K_FORK=1 ./scripts/l2_cuda_full_gate.sh`. **Vendor merge:** blocked until fork wins ≥2/3 on both platforms + qwen35 ggml smoke unchanged.

---

## L2 exit gate (not done until measured)

Compare **same model**, **same `num_ctx`**, stock vs fork:

1. Decode tok/s (prefill + generate)
2. Peak VRAM / unified memory at target ctx
3. MTP acceptance rate (when `--spec-type` configured)
4. Mac Metal + CUDA 5080 sign-off scripts pass

**Pass criteria to merge vendor:** fork wins on **≥2 of 3** (tok/s, max ctx, VRAM) on **both** 5080 and M-series without regressing qwen35 compat smoke.

---

## Cherry-pick order (if merging)

1. GGML type slots + `--cache-type-*` whitelist (`ggml.h`, `common/arg.cpp`)
2. CPU QJL + Polar decode (`ggml-cpu/qjl/*`, `quants-polar.c`)
3. CUDA kernels (`ggml-cuda/qjl.*`, `polarquant.*`, `turboquant.*`)
4. Metal `eliza-shipped/` kernels
5. `--ctx-checkpoints` server hooks
6. MTP / fused attn (optional; overlaps voice L7)

Re-apply or drop zerollama **Ollama patches** per file conflict — see [ggml-b9509-migration.md](./ggml-b9509-migration.md).

---

## Non-goals (L2)

- Replacing Go ggml vendor in one PR
- Eliza voice engine-bridge / OmniVoice monolith
- AOSP / Capacitor build paths
