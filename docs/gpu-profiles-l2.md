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

**In-process ggml** uses the same **elizaOS base + 16 Ollama patches** as the runtime sibling (`vendor/llama-cpp-c84b3020/` → sync). Rebase: `./scripts/vendor/rebase_vendor_unified.sh --sync`.

---

## Unified runtime pin

| Field | Value |
|-------|--------|
| Repo | `https://github.com/elizaOS/llama.cpp.git` |
| Ref (Jun 2026) | `c84b30200c8d512c00c9d61c96bed078f1c0024d` (`LLAMA_CPP_COMMIT`) |
| Sibling tree | `../llama.cpp` (default) |
| Build | `./scripts/build/build_llama_server.sh` |

**Delta `96dd1a8` → `c84b302` (35 commits):** mostly voice/mobile (Kokoro TTS, OmniVoice FFI, Silero VAD, Android Vulkan). **No changes** to QJL/Polar/TBQ kernels or `dflash-draft` in that range — L2 bench numbers should match prior pin unless Vulkan shader paths matter on your host.

**Already on stock b9781 (no fork patch needed):** `--ctx-checkpoints` ([ggml-org#15293](https://github.com/ggml-org/llama.cpp/pull/15293)), SWA checkpoint cell filter ([ggml-org#23981](https://github.com/ggml-org/llama.cpp/pull/23981)).

**Still fork-only vs stock b9781:** `--checkpoint-every-n-tokens`, QJL/Polar/TBQ KV types, `dflash-draft` architecture (required for `eliza-1-*-dflash` models).

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
| `--checkpoint-every-n-tokens N` | Token interval between ctx checkpoints (fork; was `--ctx-checkpoint-interval` in eliza JSON key) |

**Detection:** fork `llama-server --help` includes `--checkpoint-every-n-tokens` and custom cache types (`qjl1_256`, …). Stock **b9781** has `--ctx-checkpoints` upstream but not `--checkpoint-every-n-tokens`.

---

## Build & run (runtime path)

### macOS / Metal

```bash
# 1. Build unified sibling (Metal)
./scripts/build/build_llama_server.sh

# 2. Point runtime at binary; fork profiles auto-probe from --help
export LLAMA_CPP_ROOT=$PWD/../llama.cpp
export LLAMA_SERVER_BIN=$LLAMA_CPP_ROOT/build/bin/llama-server
export LLAMA_CPP_LIB=$LLAMA_CPP_ROOT/build/bin/libllama.dylib
export ZEROLLAMA_LLAMA_FORK=1   # or omit: auto-probes when binary set

# 3. Smoke probe + profile argv
./scripts/phase/l2_fork_eval.sh

# 4. Metal A/B benchmark (L1 vs fork profiles, same binary)
M3_LLAMA_MODEL=/path/to/model.gguf ./scripts/phase/l2_metal_bench.sh
# Output: L2_METAL_BENCH_OUT=/tmp/l2-metal-bench.json (default)

# 5. Runtime subprocess compat (L1 + fork profile legs)
./scripts/phase/l2_runtime_compat_smoke.sh

# 6. Full gate (eval + compat + bench + verdict)
L2_RUN_27K=1 L2_RUN_131K_FORK=1 ./scripts/phase/l2_full_gate.sh
# 131k leg: fork-only, L2_HIGH_CTX_WARMUPS=2 decode warmups before timed runs

# 7. Serve / sign-off
./scripts/phase/m3_metal_signoff.sh
```

### Linux / CUDA (RTX 5080-class)

```bash
# 1. Build unified sibling (CUDA)
./scripts/build/build_llama_server.sh
# Headless container: LLAMA_BUILD_WEBUI=OFF (default on Linux via build script)

# 2. Point runtime at binary
export LLAMA_CPP_ROOT=$PWD/../llama.cpp
export LLAMA_SERVER_BIN=$LLAMA_CPP_ROOT/build/bin/llama-server
export LLAMA_CPP_LIB=$LLAMA_CPP_ROOT/build/bin/libllama.so    # .so on Linux
export ZEROLLAMA_LLAMA_FORK=1

# 3. Smoke probe + profile argv
./scripts/phase/l2_fork_eval.sh

# 4. CUDA A/B benchmark (L1 vs fork profiles, same binary)
CUDA_LLAMA_MODEL=/path/to/model.gguf ./scripts/phase/l2_cuda_bench.sh
# Output: L2_CUDA_BENCH_OUT=/tmp/l2-cuda-bench.json (default)

# 5. Runtime compat smoke (Linux variant — uses linux_runtime_serve_lib + .so)
CUDA_LLAMA_MODEL=/path/to/model.gguf ./scripts/phase/l2_cuda_runtime_compat_smoke.sh

# 6. Full CUDA gate (eval + compat + bench legs + verdict)
L2_RUN_27K=1 L2_RUN_131K_FORK=1 ./scripts/phase/l2_cuda_full_gate.sh
# 131k leg: fork-only, L2_HIGH_CTX_WARMUPS=2 decode warmups before timed runs

# 7. Serve / sign-off
./scripts/gpu/gpu_5080_session.sh
```

**Env:**

| Variable | Effect |
|----------|--------|
| `ZEROLLAMA_LLAMA_FORK=1` | Force fork profile merge (QJL/Polar, checkpoints) |
| `ZEROLLAMA_LLAMA_FORK=0` | Force stock sanitize (L1 default) |
| *(unset)* | Auto: probe `LLAMA_SERVER_BIN --help` for `--ctx-checkpoints` |
| `LLAMA_CPP_ROOT` / `LLAMA_CPP_COMMIT` | Override clone path / commit (`ensure_llama_cpp_sibling.sh`) |

---

## Runtime integration (shipped)

| Component | Role |
|-----------|------|
| `runtime/llama_fork.py` | Fork detection (env + binary probe) |
| `runtime/gpu_profiles.py` | `_eliza_fork_llama_server_flags` merge; emit `--ctx-checkpoints` when present |
| `/health` | `llama_fork` object + `gpu_profile.llama_fork` |
| `scripts/build/build_llama_server.sh` | Unified runtime build (elizaOS base @ LLAMA_CPP_COMMIT) |
| `scripts/build/build_eliza_llama_server.sh` | Deprecated alias → `build_llama_server.sh` |
| `scripts/phase/l2_fork_eval.sh` | Probe + pytest smoke |
| `scripts/phase/l2_metal_bench.sh` | Darwin A/B: L1 vs fork profiles (same binary) |
| `scripts/phase/l2_cuda_bench.sh` | Linux/CUDA A/B: L1 vs fork profiles (same binary) |
| `scripts/phase/l2_runtime_compat_smoke.sh` | Darwin subprocess compat: L1 + fork profile legs |
| `scripts/phase/l2_cuda_runtime_compat_smoke.sh` | Linux subprocess compat: mirrors compat smoke with `.so` + `linux_runtime_serve_lib` |
| `scripts/phase/l2_gate_report.sh` | Verdict from one or more bench JSON files |
| `scripts/phase/l2_full_gate.sh` | Darwin gate orchestrator: eval + compat + bench legs + report |
| `scripts/phase/l2_cuda_full_gate.sh` | CUDA gate orchestrator: same structure as Metal gate |
| `scripts/runtime/linux_runtime_serve_lib.sh` | Shared sidecar start/stop helpers for Linux (mirrors `macos_runtime_serve_lib.sh`) |

**WHY sibling tree first:** `vendor/llama-cpp-b9781/` + `llama/patches/` stay on b9781 until the gate passes.

---

## M-series sign-off (Jul 2026, M4 Max 128 GiB, vendor `8f114a9b`)

| Model | ctx | Stock | Fork | Notes |
|-------|-----|-------|------|-------|
| eliza-1-2b | 8192 | **67.2 tok/s**, q8_0 | 53.1 tok/s, tbq4_0/tbq3_0 | Stock wins decode (~21%); TBQ SET_ROWS on Metal (patch 0068) |
| eliza-1-2b (Jun 2026 ref) | 8192 | **37.6 tok/s**, q8_0 | 20.5 tok/s, tbq | Prior pin; numbers not comparable to Jul vendor rebuild |

JSON: `/tmp/l2-metal-bench.json`. **Runtime compat:** PASS (stock + fork subprocess generate on lab ports).

**Gate status (Metal):** **FAIL merge** at 8k (stock faster). **Default stays L1** (`ZEROLLAMA_LLAMA_FORK=0`). Fork is opt-in for long-ctx KV experiments.

### Patches that closed the Metal fork load gap

| Patch | What |
|-------|------|
| **0067** | Whitelist TBQ/QJL/Polar on `--cache-type-k/v`; checkpoint argv alias |
| **0068** | Metal SET_ROWS for TBQ3/TBQ4 (embedded metallib); CPU `type_traits` `from_float`; TBQ4_0 flash-attn dequant cast; ane `@rpath` stage |

**WHY 0068:** ggml abort `SET_ROWS` on Metal-preallocated `cache_k_*` when fork types lacked Metal encode kernels.

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

**Gate status (CUDA 5080):** **FAIL merge** @ 8k and **27k** (stock faster; `l2_cuda_full_gate.sh` exit 1 = verdict fail, not broken run; no long-ctx fork win on measured 9B legs). **131k fork-only:** not completed on 5080 — VRAM (9B) + QJL/model head mismatch (1B). **Vendor merge:** blocked until fork wins ≥2/3 on **both** Metal and CUDA without qwen35 regression. Fork pin **`c84b302`**; profile checkpoint argv uses `--checkpoint-every-n-tokens` (not deprecated `--ctx-checkpoint-interval`).

---

## L2 exit gate (measured Jul 2026)

Compare **same model**, **same `num_ctx`**, stock vs fork:

1. Decode tok/s (prefill + generate) — **Metal 8k: stock wins**
2. Peak VRAM / unified memory at target ctx — **estimate still favors stock** (TBQ not fully modeled)
3. MTP acceptance rate (when `--spec-type` configured) — not required for exit
4. Mac Metal + CUDA 5080 sign-off scripts pass — **Metal PASS run; merge FAIL**

**Pass criteria to flip defaults:** fork wins on **≥2 of 3** (tok/s, max ctx, VRAM) on **both** 5080 and M-series without regressing qwen35 compat smoke.

**Exit (Jul 2026):** L2 **infrastructure Done** on unified pin; **defaults stay L1**. Operators opt into fork with `ZEROLLAMA_LLAMA_FORK=1`.

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
