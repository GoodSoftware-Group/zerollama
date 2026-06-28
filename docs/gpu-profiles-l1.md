# L1 — Per-GPU llama-server profiles (autotune)

**Audience:** Operators tuning inference throughput on NVIDIA CUDA hosts and Apple Silicon Macs.

**Related:** [ROADMAP — borrowings L1](./ROADMAP.md#local-voice--llama-borrowings-eliza-v3), [phase13-runtime-vram.md](./phase13-runtime-vram.md) (estimate/clamp — complementary), [gpu-5080-operator-guide.md](./gpu-5080-operator-guide.md) (CUDA gate), [apple-silicon-metal.md](./apple-silicon-metal.md) (Metal tiers), [runtime/docs/OPERATIONS.md](../runtime/docs/OPERATIONS.md) (`/health` fields).

---

## Why L1 exists

Phase 13 answers **“will this GGUF fit at this `num_ctx`?”** L1 answers **“given hardware that fits, which llama-server flags maximize throughput?”**

**Why not one YAML for everyone:** batch size, parallel slots, flash-attn, and MTP draft depth that work on an RTX 5090 OOM or thrash on a 16 GiB 5080. Apple Silicon adds a second axis — **unified memory** is not discrete VRAM; the same `-b 4096` that helps CUDA can starve the OS on a 16 GiB MacBook Air.

**Why port eliza-v3 JSON instead of hand-tuning YAML:** eliza’s `native/configs/gpu/` already encodes measured `-b`, `-ub`, `-np`, and draft ranges per card class. Zerollama keeps **stock llama.cpp** today — fork-only KV types (QJL, Polar, TurboQuant) and argv flags (`--ctx-checkpoints`) are **stored for L2** but never emitted until the fork lands.

**Why merge at config load, not in Go ggml:** the Python runtime (inprocess or subprocess `llama-server`) owns Phase 12–15 admission and multiseq. Go ggml Metal uses a separate scheduler path; L1 targets **runtime-routed** inference first — where operators already hit Phase 13 snapshots and sign-off scripts.

---

## How it works

```text
RuntimeConfig.from_file(yaml)
        │
        ▼
maybe_apply_gpu_profile()     ← ZEROLLAMA_GPU_PROFILE, llama_profile in YAML
        │
        ├── apple_silicon.yaml ──► hw.memsize tier (16g … 128g)
        ├── single_gpu.yaml    ──► nvidia-smi name, else VRAM bucket
        └── dual_4090.yaml     ──► NVIDIA only (never Apple tier on Linux tests)
        │
        ▼
RuntimeConfig.llama_server_args()  →  -sm -mg -np … + profile -b -ub -fa …
        │
        ▼
/health.gpu_profile + llama_args (observability)
```

**Why config-path gates platform:** on macOS, `autoconfig` prefers `apple_silicon.yaml`, but tests and explicit `ZEROLLAMA_RUNTIME_CONFIG=single_gpu.yaml` must not accidentally apply Metal tiers — `_select_profile_for_config()` ties profile family to the YAML file name.

---

## Profile files

| Path | Purpose |
|------|---------|
| `runtime/configs/gpu/index.json` | NVIDIA VRAM buckets + Apple unified-memory buckets |
| `runtime/configs/gpu/*.json` | Per-card `llama_server_flags`, optional `_eliza_fork_*`, optional `runtime_kv` |
| `runtime/runtime/gpu_profiles.py` | Detection, sanitization, argv emission |
| `runtime/runtime/nvidia_probe.py` | Shared cached `nvidia-smi` probes (autoconfig + profiles) |

### NVIDIA selection

1. **Name match** — `match_names` in JSON vs `nvidia-smi --query-gpu=name` (e.g. `RTX 4090` → `4090.json`).
2. **VRAM bucket** — if name unknown, total VRAM from `nvidia-smi` maps through `fallback_buckets` in `index.json` (e.g. 16 GiB → `rtx-5080` profile).

### Apple Silicon selection

**Why RAM tiers, not chip name:** M2 Pro and M4 Max can both ship with 48 GiB or 128 GiB — `hw.memsize` is the stable signal for KV + parallel headroom.

| Unified RAM | Profile id | Typical `-np` | Default `-c` |
|-------------|------------|---------------|--------------|
| ≤16 GiB | `apple-silicon-16g` | 1 | 16384 |
| ≤24 GiB | `apple-silicon-24g` | 1 | 32768 |
| ≤48 GiB | `apple-silicon-48g` | 2 | 32768 |
| ≤96 GiB | `apple-silicon-96g` | 4 | 65536 |
| >96 GiB | `apple-silicon-128g` | 8 | 131072 |

**128g `-c`:** tuned on M4 Max Jun 2026 sign-off where Phase 13 `suggested_max_num_ctx` reported 131072 with headroom — not a generic default for all >96 GiB machines until measured.

**128g `runtime_kv`:** profile sets `num_blocks_per_device: 8192` (×16 = 131072 tokens) so Python PA admission matches llama-server `-c 131072`. Env `ZEROLLAMA_KV_NUM_BLOCKS` still wins.

---

## Stock llama.cpp safety

| eliza / fork feature | L1 behavior | Why |
|----------------------|-------------|-----|
| `qjl1_256`, `q4_polar`, `turbo*` cache types | Rewritten to `q8_0` at load | Stock ggml rejects unknown cache types |
| `ctx_checkpoints`, `ctx_checkpoint_interval` | Kept in `_fork_only_llama_server_flags`; stripped from argv | Would crash stock `llama-server` with unknown flag |
| `mlock: true` in eliza JSON | **`false` in all shipped profiles** | Pinning weights fails in containers and shared Mac RAM; opt-in via env |
| `n_gpu_layers: 999` | Mapped to `-1` (all layers) | eliza convention vs llama.cpp `-1` |

---

## Operator controls

| Variable / YAML | Effect | Why |
|-----------------|--------|-----|
| `ZEROLLAMA_GPU_PROFILE=0` | Disable autotune; YAML `llama_parallel_slots` only | A/B against profile or broken detection |
| `ZEROLLAMA_GPU_PROFILE_CTX=0` | Do not emit profile `-c` | Long-context models; avoid silent cap below model training ctx |
| `ZEROLLAMA_GPU_PROFILE_MLOCK=1` | Allow `--mlock` when profile sets `mlock: true` | Dedicated bare-metal hosts only |
| `llama_profile.apply_ctx_size: false` | Same as `ZEROLLAMA_GPU_PROFILE_CTX=0` | Persistent opt-out in YAML |
| `llama_profile.enabled: false` | Disable profiles | Same as env off |
| `LLAMA_SERVER_EXTRA_ARGS` | Appended **after** profile flags | Operator wins on duplicate flags (e.g. `-c 262144`) |

---

## Verify

### `/health` (runtime `:8081`)

```bash
curl -s http://127.0.0.1:8081/health | jq '{
  gpu_profile,
  llama_parallel_slots: .parallel_slots_default,
  llama_args
}'
```

Expect `gpu_profile.id`, `bucket_label`, and on Mac `unified_memory_gb`.

### Unit tests

```bash
cd runtime && uv run pytest tests/test_gpu_profiles.py -q
```

### CUDA 5080 calibration (ship hardware)

**Why not trust eliza port values:** 4090 profile scaled to 16 GiB kept `-np 4 -b 2048`; on a 1B smoke model that regressed single-stream decode **−12.5%** vs profile OFF. Production-sized GGUFs (9B) only showed **−1%** at the same flags — the bug is slot overhead on tiny models, not broken detection.

**Script:** `./scripts/l1_cuda_calibrate.sh` — profile OFF baseline vs ON (+ optional `L1_SWEEP_NP=1,2,4`).

**WHY calibrate env (automatic in script):** `ZEROLLAMA_LLAMA_FORK=0` (stock `q8_0` only — fork QJL is **L2**); `ZEROLLAMA_GPU_PROFILE_CTX=0` (do not emit profile `-c` during 8k bench — `-c 32768` falsely regresses single-stream **~−39%**). Concurrent bench keeps `ZEROLLAMA_GPU_PROFILE_CTX=1` for realistic serve.

```bash
export CUDA_LLAMA_MODEL=/root/eliza-1-9b-256k.gguf   # 7B–9B class on 16GB
5080_stop_serve                                      # WHY: embedded serve races uv sidecar on :8081
./scripts/l1_cuda_calibrate.sh
# Sweep parallel slots after single-stream baseline:
L1_SWEEP_NP=1,2,4 CUDA_LLAMA_MODEL=/root/your-prod.gguf ./scripts/l1_cuda_calibrate.sh
```

`l2_cuda_bench.sh` honors `ZEROLLAMA_GPU_PROFILE=0|1` (default `1`) for OFF/ON legs.

**Jun 2026 tuned `rtx-5080.json` (CT 1564):** `n_parallel=2`, `batch_size=1024`, `ubatch_size=256` (half 4090 batch for 16 GiB). Re-measure after edits.

| Model | ctx | OFF tok/s | ON tok/s | Δ |
|-------|-----|-----------|----------|---|
| OuteTTS 1B Q8 | 8192 | 43.48 | 43.69 | **+0.5%** |
| eliza-1 9B | 8192 | **~90** (`np=1`) | **~85** (`np=2`, no profile `-c`) | **−5%** single-stream |

**Concurrent bench:** `./scripts/l1_cuda_concurrent_bench.sh` — fires `L1C_N` parallel `/api/generate` requests simultaneously and measures aggregate tok/s and per-thread wall time, A/B profile OFF vs ON. This is the **production closure** for `n_parallel=2`: single-stream may show small **−5%** np overhead on one request; the win appears under concurrency where two slots amortise prefill.

```bash
# Default: n_concurrent=2 (matches n_parallel), 9B class model
CUDA_LLAMA_MODEL=/root/eliza-1-9b-256k.gguf ./scripts/l1_cuda_concurrent_bench.sh

# Sweep n_parallel values while N=4 concurrent:
L1C_N=4 L1C_SWEEP_NP="1,2,4" CUDA_LLAMA_MODEL=/root/eliza-1-9b-256k.gguf ./scripts/l1_cuda_concurrent_bench.sh
```

The summary prints aggregate tok/s (sum across all threads) and `%` vs OFF. PASS when ON ≥ OFF at the target concurrency.

**Jun 2026 concurrent (CT 1564, eliza-1 9B, `L1C_N=2`, ctx 8192, predict 128, vendor `llama-server` + matching `libllama.so`):**

| Leg | agg tok/s | errors | vs OFF |
|-----|-----------|--------|--------|
| profile OFF (`n_parallel=1`) | **~55** | 1×502 on 2nd thread (expected) | — |
| profile ON (`rtx-5080`, `n_parallel=2`) | **~65** | 0 | **+~16–20%** |

Artifact: `/tmp/l1-production-gate/concurrent/profile-on-default.json` (reruns ±1–2 tok/s).

### Sign-off gates

| Platform | Script | L1 validation |
|----------|--------|----------------|
| Apple Silicon | `./scripts/l1_metal_gate.sh` or `./scripts/m3_metal_signoff.sh` | RAM-tier pytest + optional live `/health.gpu_profile` |
| CUDA 5080-class | `./scripts/l1_cuda_full_gate.sh` | Single-stream calibrate + concurrent N=2; merged `gate.json` + verdict |
| Full 5080 session | `RUN_E2E_L1=1 ./scripts/gpu_5080_session.sh` | Phase 10–13 snapshot + L1 CUDA gate when model set |

**Production gate (CUDA):**

```bash
export CUDA_LLAMA_MODEL=/root/eliza-1-9b-256k.gguf   # ship proxy on 16GB; supernova optional re-run
./scripts/l1_cuda_full_gate.sh
# Artifacts: /tmp/l1-production-gate/{calibrate,concurrent}/ + gate.json
./scripts/l1_gate_report.sh /tmp/l1-production-gate/gate.json
```

**Pass criteria:**

| Leg | Threshold | Jun 2026 (CT 1564, eliza-1 9B) |
|-----|-----------|--------------------------------|
| Single-stream | ON ≥ OFF (non-regression @ 0%) | **−5%** @ 8k — np=2 slot overhead on one stream; informational unless tuning `rtx-5080.json` |
| Concurrent `L1C_N=2` | ON **>** OFF (strict win) | **+~16–20%** agg tok/s (~65 vs ~55) — **production PASS** |

Optional supernova-class re-validation when that GGUF is on host — not blocking L1 Done.

**Metal gate:**

```bash
./scripts/l1_metal_gate.sh
# or platform dispatcher:
./scripts/l1_full_gate.sh
```

---

## Status (Jun 2026)

| Platform | Status | Notes |
|----------|--------|-------|
| **Apple Silicon** | **Done** — RAM tiers, M4 Max 128g sign-off, `l1_metal_gate.sh` | Re-measure after L2 fork if KV types change |
| **NVIDIA CUDA** | **Done (concurrent)** — `rtx-5080.json` tuned; `l1_cuda_full_gate.sh` concurrent **PASS** on eliza-1 9B; single-stream **−5%** @ 8k (np overhead) | Optional: re-run gate on supernova-class GGUF; tune batch if single-stream regression matters |

**Not in L1:** Go ggml Metal runner flags (separate scheduler); voice phrase cache (**L5**); eliza fork kernels (**L2** — see [gpu-profiles-l2.md](./gpu-profiles-l2.md)).

---

## L2 pointer

When `ZEROLLAMA_LLAMA_FORK=1` or the `llama-server` binary advertises `--ctx-checkpoints`, profiles merge `_eliza_fork_llama_server_flags` (QJL/Polar cache types) and emit checkpoint argv. Build with `./scripts/build_llama_server.sh` (unified elizaOS tree). Full evaluation gate: [gpu-profiles-l2.md](./gpu-profiles-l2.md).
