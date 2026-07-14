# L2 — elizaOS/llama.cpp fork evaluation

**Audience:** Contributors evaluating TurboQuant / QJL / PolarQuant profile wins on the **unified** `llama-server` binary.

**Related:** [gpu-profiles-l1.md](./gpu-profiles-l1.md), [ROADMAP — L2](./ROADMAP.md#local-voice--llama-borrowings-eliza-v3), [phase17-llama-server.md](./phase17-llama-server.md), [runtime/LLAMA_CPP_PIN.md](../runtime/LLAMA_CPP_PIN.md).

---

## One binary, two profile modes

Zerollama builds **one** `llama-server` from **`elizaOS/llama.cpp`** @ `LLAMA_CPP_COMMIT`. That tree is a superset of stock ggml-org: dflash-draft, QJL/Polar/TBQ, checkpoints, and upstream fixes.

L2 gates compare **profile modes on the same binary**, not stock vs fork siblings:

| Leg | `ZEROLLAMA_LLAMA_FORK` | Cache types |
|-----|------------------------|-------------|
| L1 (stock profile) | `0` | `q8_0` / tuned flags |
| Fork profile | auto or `1` | `qjl1_256`, `q4_polar`, TBQ, checkpoints |

**Deprecated:** separate `../eliza-llama.cpp` checkout and `build_eliza_llama_server.sh` (now wraps `build_llama_server.sh`).

---

## Why L2 still exists

L1 ships **tuned flags on q8_0** by default. Eliza-v3’s largest wins — **QJL K-cache**, **PolarQuant V-cache**, **TurboQuant** — need fork cache types enabled at runtime. L2 measures whether fork profiles beat L1 on your GPU before we flip defaults.

**In-process ggml** shares the same **ggml-org pin + Ollama/zerollama patches** as the runtime vendor (`vendor/llama-cpp-<FETCH_HEAD>/` → sync). Rebase: `./scripts/rebase_vendor_unified.sh --sync`.

---

## Unified runtime pin

| Field | Value |
|-------|--------|
| Repo (runtime + in-process) | `https://github.com/ggml-org/llama.cpp.git` |
| Ref (Jul 2026) | `8f114a9b…` (`LLAMA_CPP_COMMIT` / `Makefile.sync` `FETCH_HEAD`) |
| QJL/Polar/TBQ | Patches **0026–0030** (+ follow-ups) on the ggml-org vendor tree — not a separate elizaOS checkout |
| Sibling tree | `../llama.cpp` optional; prefer `vendor/llama-cpp-<pin>/` when built |
| Build | `./scripts/build_llama_server.sh` |

**Still measured as L2:** enabling fork **profiles** (`ZEROLLAMA_LLAMA_FORK` → `qjl1_256` / `q4_polar`) vs L1 `q8_0`. Kernel extraction ≠ default-on.

**Legacy note:** Jun 2026 gates ran on elizaOS `@ c84b3020`. That tree may still exist as a built fallback when `vendor/llama-cpp-8f114a9b` is not materialised — pin status: `./scripts/phase17_l2_pin_status.sh`.

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

**Flash Attention required:** quantized fork KV (`tbq*`, `qjl*`, `q4_polar`) needs `-fa on`. Without it llama-server fails with *quantized V cache was requested, but this requires Flash Attention*. GPU JSON already sets `flash_attn: true` on CUDA cards; `llama_argv_from_profile_flags` also forces `-fa on` when those cache types are active.

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
| `--checkpoint-every-n-tokens N` | Token interval between ctx checkpoints (fork; was `--ctx-checkpoint-interval` in eliza JSON key) |

**Detection:** fork `llama-server --help` includes `--checkpoint-every-n-tokens` and custom cache types (`qjl1_256`, …). Stock **b9781** has `--ctx-checkpoints` upstream but not `--checkpoint-every-n-tokens`.

---

## Build & run (runtime path)

### macOS / Metal

```bash
# 1. Build unified sibling (Metal)
./scripts/build_llama_server.sh

# 2. Point runtime at binary; fork profiles auto-probe from --help
export LLAMA_CPP_ROOT=$PWD/../llama.cpp
export LLAMA_SERVER_BIN=$LLAMA_CPP_ROOT/build/bin/llama-server
export LLAMA_CPP_LIB=$LLAMA_CPP_ROOT/build/bin/libllama.dylib
export ZEROLLAMA_LLAMA_FORK=1   # or omit: auto-probes when binary set

# 3. Smoke probe + profile argv
./scripts/l2_fork_eval.sh

# 4. Metal A/B benchmark (L1 vs fork profiles, same binary)
M3_LLAMA_MODEL=/path/to/model.gguf ./scripts/l2_metal_bench.sh
# Output: L2_METAL_BENCH_OUT=/tmp/l2-metal-bench.json (default)

# 5. Runtime subprocess compat (L1 + fork profile legs)
./scripts/l2_runtime_compat_smoke.sh

# 6. Full gate (eval + compat + bench + verdict)
L2_RUN_27K=1 L2_RUN_131K_FORK=1 ./scripts/l2_full_gate.sh
# 131k leg: fork-only, L2_HIGH_CTX_WARMUPS=2 decode warmups before timed runs

# 7. Serve / sign-off
./scripts/m3_metal_signoff.sh
```

### Linux / CUDA (RTX 5080-class)

```bash
# 1. Build unified sibling (CUDA)
./scripts/build_llama_server.sh
# Headless container: LLAMA_BUILD_WEBUI=OFF (default on Linux via build script)

# 2. Point runtime at binary
export LLAMA_CPP_ROOT=$PWD/../llama.cpp
export LLAMA_SERVER_BIN=$LLAMA_CPP_ROOT/build/bin/llama-server
export LLAMA_CPP_LIB=$LLAMA_CPP_ROOT/build/bin/libllama.so    # .so on Linux
export ZEROLLAMA_LLAMA_FORK=1

# 3. Smoke probe + profile argv
./scripts/l2_fork_eval.sh

# 4. CUDA A/B benchmark (L1 vs fork profiles, same binary)
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
| `ZEROLLAMA_LLAMA_FORK=1` | Force fork profile merge (checkpoints + KV types) |
| `ZEROLLAMA_LLAMA_FORK=0` | Force stock sanitize (L1 default) |
| *(unset)* | Auto: probe `LLAMA_SERVER_BIN --help` for fork KV markers |
| `ZEROLLAMA_LLAMA_FORK_PROFILE=vram` | **Default** when fork on — TBQ (`_eliza_fork_vram_*`) |
| `ZEROLLAMA_LLAMA_FORK_PROFILE=speed` | QJL/Polar (`_eliza_fork_*`) — experimental; large CUDA tok/s hit |
| `LLAMA_CPP_ROOT` / `LLAMA_CPP_COMMIT` | Override clone path / commit (`ensure_llama_cpp_sibling.sh`) |

---

## Runtime integration (shipped)

| Component | Role |
|-----------|------|
| `runtime/llama_fork.py` | Fork detection (env + binary probe) |
| `runtime/gpu_profiles.py` | `_eliza_fork_*` / `_eliza_fork_vram_*` merge via `ZEROLLAMA_LLAMA_FORK_PROFILE`; emit `--ctx-checkpoints` when present |
| `/health` | `llama_fork` object + `gpu_profile.llama_fork` + `llama_fork_profile` |
| `scripts/build_llama_server.sh` | Unified runtime build (elizaOS base @ LLAMA_CPP_COMMIT) |
| `scripts/build_eliza_llama_server.sh` | Deprecated alias → `build_llama_server.sh` |
| `scripts/l2_fork_eval.sh` | Probe + pytest smoke |
| `scripts/l2_metal_bench.sh` | Darwin A/B: L1 vs fork profiles (same binary) |
| `scripts/l2_cuda_bench.sh` | Linux/CUDA A/B: L1 vs fork profiles (same binary) |
| `scripts/l2_runtime_compat_smoke.sh` | Darwin subprocess compat: L1 + fork profile legs |
| `scripts/l2_cuda_runtime_compat_smoke.sh` | Linux subprocess compat: mirrors compat smoke with `.so` + `linux_runtime_serve_lib` |
| `scripts/l2_gate_report.sh` | Verdict from one or more bench JSON files |
| `scripts/l2_full_gate.sh` | Darwin gate orchestrator: eval + compat + bench legs + report |
| `scripts/l2_cuda_full_gate.sh` | CUDA gate orchestrator: same structure as Metal gate |
| `scripts/linux_runtime_serve_lib.sh` | Shared sidecar start/stop helpers for Linux (mirrors `macos_runtime_serve_lib.sh`) |

**WHY sibling tree first:** `vendor/llama-cpp-b9781/` + `llama/patches/` stay on b9781 until the gate passes.

---

## M-series sign-off (Jun 2026, M4 Max 128 GiB)

| Model | ctx | Stock | Fork | Notes |
|-------|-----|-------|------|-------|
| eliza-1-2b | 8192 | **37.6 tok/s**, q8_0 | 20.5 tok/s, tbq4_0/tbq3_0 | Stock wins decode + VRAM est |
| eliza-1-27b | 26624 | 13.2 tok/s, q8_0 | 12.7 tok/s, tbq | Stock wins decode (~4%); VRAM est heuristic favors stock (TBQ not modeled) |
| eliza-1-27b | 131072 | Rejected (KV est) | **5.0 tok/s** | Fork-only; `ZEROLLAMA_GPU_PROFILE_CTX=0` + `runtime_kv` 8192 blocks |

JSON: `/tmp/l2-metal-bench.json`, `/tmp/l2-gate/`. **Runtime compat smoke:** PASS (stock + fork subprocess generate).

**Gate status (Metal):** **FAIL merge** at small ctx (stock faster). Fork may still win **max ctx + VRAM** at 27k+ — run `L2_RUN_27K=1 L2_RUN_131K_FORK=1 ./scripts/l2_full_gate.sh`.

### CUDA 5080 sign-off (Jun 2026, CT 1564, RTX 5080)

| Model | ctx | Stock | Fork | Notes |
|-------|-----|-------|------|-------|
| OuteTTS 1B Q8 | 8192 | **79.3 tok/s**, q8_0/q8_0 | 56.9 tok/s, qjl1_256/q4_polar | Stock wins decode (~28%); VRAM est tied; reruns ±1 tok/s |
| eliza-1 9B | 8192 | **18.6 tok/s**, q8_0/q8_0 | 14.4 tok/s, qjl1_256/q4_polar | Stock wins (~23%); same model long-ctx gate |
| eliza-1 9B | 26624 | **18.5 tok/s**, q8_0/q8_0 | 14.3 tok/s, qjl1_256/q4_polar | **No fork salvage at 27k** — delta ~same as 8k |
| eliza-1 9B | 131072 fork | — | **blocked** | Admission: ~31 GiB required on 16 GiB 5080 |
| OuteTTS 1B Q8 | 131072 fork | — | **blocked** | QJL `qjl1_256` incompatible with `n_embd_head_k=64` on this GGUF |

JSON: `/tmp/l2-cuda-gate/bench-8k.json` (1B), `/tmp/l2-cuda-gate-long/bench-{8k,27k}.json` (9B). **131k:** needs `ZEROLLAMA_KV_NUM_BLOCKS=8192` + GGUF compatible with QJL head dims; 9B needs >16 GiB VRAM at 131k.

**Fork build footgun (container):** eliza sibling may need `-DLLAMA_BUILD_WEBUI=OFF` — default WebUI asset download fails headless.

**Gate status (CUDA 5080):** **FAIL merge** @ 8k and **27k** (stock faster; `l2_cuda_full_gate.sh` exit 1 = verdict fail, not broken run; no long-ctx fork win on measured 9B legs). **131k fork-only:** not completed on 5080 — VRAM (9B) + QJL/model head mismatch (1B). **Vendor profile defaults:** blocked until fork wins ≥2/3 on **both** Metal and CUDA without qwen35 regression. Kernels: patches **0026–0030** + CUDA follow-ups **0067–0070** on ggml-org `8f114a9b`; `FORK=1` defaults to TBQ (`FORK_PROFILE=vram`). Checkpoint argv uses `--checkpoint-every-n-tokens`.

### CUDA 4090 exploratory (Jul 2026, dual RTX 4090)

Gate fixes required for valid A/B: pin `single_gpu.yaml` into `linux_runtime_start_sidecar` (empty config re-enabled autoconfig → `dual_4090` + `serve.llama_fork: stock`); force `ZEROLLAMA_LLAMA_FORK=1` on fork leg; sample live `nvidia-smi`.

| Model | ctx | Stock | Fork (TBQ) | Notes |
|-------|-----|-------|------------|-------|
| llama3.2 3B | 8192 | **154 tok/s**, 4190 MiB | 133 tok/s, 3936 MiB | Decode **−14%**; VRAM **−6%** |
| llama3.2 3B | 26624 | **99.5 tok/s**, 5262 MiB | 98.7 tok/s, 4440 MiB | Decode ~flat; VRAM **−16%** |
| llama3.2 3B | 65536 | **93.0 tok/s**, 7522 MiB | 88.6 tok/s, 5504 MiB | Decode **−5%**; VRAM **−27%** |
| llama3.2 3B | 131072 | **91.6 tok/s**, 11470 MiB | 85.7 tok/s, 7436 MiB | Decode **−6%**; VRAM **−35%** (~4 GiB saved) |

**8f114a9b re-gate (Jul 2026, same host, TBQ via `L2_FORK_CACHE_TYPE_*`):** stock still wins decode; fork wins VRAM. Short legs (contended host): `/tmp/l2-cuda-gate-8f114a9b-llama32-tbq/`. Long-ctx (prod stopped): `/tmp/l2-cuda-gate-8f114a9b-llama32-long/`.

| Model | ctx | Stock | Fork (TBQ) | Notes |
|-------|-----|-------|------------|-------|
| llama3.2 3B | 8192 | 10.4 tok/s, 3562 MiB | 9.5 tok/s, 3310 MiB | Contended host; decode **−9%**; VRAM **−7%** |
| llama3.2 3B | 26624 | 34.7 tok/s, 4700 MiB | 26.1 tok/s, 3922 MiB | Contended; decode **−25%**; VRAM **−17%** |
| llama3.2 3B | 65536 | 33.4 tok/s, 7116 MiB | 26.8 tok/s, 5208 MiB | Quiet host; decode **−20%**; VRAM **−27%** |
| llama3.2 3B | 131072 | 32.1 tok/s, 11196 MiB | 25.2 tok/s, 7384 MiB | Quiet host; decode **−21%**; VRAM **−34%** |

VRAM deltas at 65k/131k match the c84 reference (−27% / −35%). Absolute tok/s is lower via the Python sidecar path than direct `llama-server`, but the VRAM tradeoff is confirmed on **`8f114a9b`**.

**QJL/Polar on llama3.2:** aborts on legacy `c84b3020`. On pin **`8f114a9b` + patches 0067–0070**, QJL/Polar **loads and runs** (no abort). Contended-host sidecar A/B (`L2_FORK_CACHE_TYPE_*=qjl1_256/q4_polar`, prod left up): `/tmp/l2-cuda-gate-8f114a9b-llama32-qjl/`.

| Model | ctx | Stock | Fork (QJL/Polar) | Notes |
|-------|-----|-------|------------------|-------|
| llama3.2 3B | 8192 | 10.8 tok/s, 3556 MiB | 5.6 tok/s, 3284 MiB | Decode **−48%**; VRAM **−8%** |
| llama3.2 3B | 26624 | 63.6 tok/s, 4696 MiB | 9.8 tok/s, 3740 MiB | Decode **−85%**; VRAM **−20%** |

Same host TBQ @ 8k/27k was only **−9% / −25%** decode — QJL **speed** profile is far worse on tok/s than TBQ **vram**. Prefer `FORK_PROFILE=vram` (TBQ) for headroom; treat `speed` (QJL/Polar) as experimental. TBQ load segfault on bare rebase was missing CPU `type_traits_cpu.from_float` (**0070**).

**Verdict:** do **not** flip defaults for tok/s. **Do** opt into fork **VRAM profile** (TBQ) when long-ctx headroom matters (agent 65k–131k, multi-slot). **Do not** default `speed`/QJL on CUDA. Artifacts: `/tmp/l2-cuda-gate-4090-llama32-tbq/`, `/tmp/l2-cuda-gate-4090-llama32-long/`, `/tmp/l2-cuda-gate-8f114a9b-llama32-tbq/`, `/tmp/l2-cuda-gate-8f114a9b-llama32-long/`, `/tmp/l2-cuda-gate-8f114a9b-llama32-qjl/`.

**When to enable fork for VRAM (operator)**

```bash
# Env (any topology) — FORK=1 defaults to vram/TBQ; set speed only for QJL experiments
export ZEROLLAMA_LLAMA_FORK=1
# export ZEROLLAMA_LLAMA_FORK_PROFILE=vram   # optional; already the default
# export ZEROLLAMA_LLAMA_FORK_PROFILE=speed  # QJL/Polar — expect large decode regression on CUDA

# Dual 4090 drop-in YAML (serve.llama_fork + llama_fork_profile)
export ZEROLLAMA_RUNTIME_CONFIG=/path/to/zerollama/runtime/configs/dual_4090_vram.yaml
```

| Goal | Setting | Why |
|------|---------|-----|
| Max decode tok/s | `ZEROLLAMA_LLAMA_FORK=0` / `dual_4090.yaml` (default) | L1 q8_0 wins measured legs |
| Fit more ctx / slots | `FORK=1` (defaults to TBQ) / `dual_4090_vram.yaml` | TBQ −27…−35% VRAM @ 65k–131k on 4090 |
| Max compression (experimental) | `FORK=1` + `FORK_PROFILE=speed` | QJL/Polar — runs on `8f` llama3.2 but **−48…−85%** decode @ 8k/27k; prefer TBQ |

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
5. `--checkpoint-every-n-tokens` + `--ctx-checkpoints` server hooks (ctx-checkpoints already in b9781; interval flag is fork-only)
6. MTP / fused attn (optional; overlaps voice L7)

Re-apply or drop zerollama **Ollama patches** per file conflict — see [ggml-b9509-migration.md](./ggml-b9509-migration.md).

---

## Non-goals (L2)

- Replacing Go ggml vendor in one PR
- Eliza voice engine-bridge / OmniVoice monolith
- AOSP / Capacitor build paths
