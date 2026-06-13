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

```bash
# 1. Build fork sibling (Metal on Mac, CUDA on Linux)
./scripts/build_eliza_llama_server.sh

# 2. Point runtime at fork binary + enable fork profile merge
export LLAMA_CPP_ROOT=$PWD/../eliza-llama.cpp
export LLAMA_SERVER_BIN=$LLAMA_CPP_ROOT/build/bin/llama-server
export LLAMA_CPP_LIB=$LLAMA_CPP_ROOT/build/bin/libllama.dylib   # .so on Linux
export ZEROLLAMA_LLAMA_FORK=1   # or omit: auto-probes --help when binary set

# 3. Smoke probe + profile argv
./scripts/l2_fork_eval.sh

# 4. Metal A/B benchmark (stock vs fork decode tok/s + VRAM estimate)
M3_LLAMA_MODEL=/path/to/model.gguf ./scripts/l2_metal_bench.sh
# Output: L2_METAL_BENCH_OUT=/tmp/l2-metal-bench.json (default)

# 5. Serve / sign-off (compare tok/s vs stock binary)
./scripts/m3_metal_signoff.sh          # Mac
# ./scripts/gpu_5080_session.sh        # CUDA
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

**WHY sibling tree first:** `vendor/llama-cpp-b9611/` + `llama/patches/` stay on b9611 until the gate passes.

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
