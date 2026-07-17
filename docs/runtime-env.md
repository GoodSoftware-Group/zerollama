# Runtime environment (Python sidecar)

**Audience:** Operators debugging L3 prefix cache, Phase 15 KV, and VRAM policy without reading twenty modules.

**Implementation:** [`runtime/runtime/env.py`](../runtime/runtime/env.py) — tri-state bools, YAML/profile presets, platform defaults. Env **always wins** over YAML when explicitly set.

---

## Prefer profiles over raw env

| Goal | Use |
|------|-----|
| Multi-slot agent + Radix | `ZEROLLAMA_L3_PROFILE=agent` |
| Custom runtime YAML | `ZEROLLAMA_RUNTIME_CONFIG=runtime/configs/….yaml` |
| L3 + prefix trace debug | `ZEROLLAMA_DEBUG=l3` |
| Phase 15 infer spans | `ZEROLLAMA_DEBUG=infer` |

Check effective values (no running server required):

```bash
./scripts/runtime/runtime_env_doctor.sh
# or live:
curl -s :8081/health | jq '.llama_cache.runtime_env, .autoconfig'

# llama.cpp patches / vendor / llama-server binary (offline):
./scripts/vendor/llama_patch_doctor.sh
# optional live route probe:
LLAMA_PATCH_PROBE_URL=http://127.0.0.1:8082 ./scripts/vendor/llama_patch_doctor.sh
```

---

## L3 prefix cache

| Variable | Default | Notes |
|----------|---------|-------|
| `ZEROLLAMA_L3_PROFILE` | — | `agent` → `l3_agent_subprocess.yaml` |
| `ZEROLLAMA_LLAMA_CACHE` | `1` | Master L3 off switch |
| `ZEROLLAMA_LLAMA_CACHE_DISK` | **smart** | Unset: off Darwin, on Linux subprocess |
| `ZEROLLAMA_RADIX_PREFIX_SHARE` | `0` / YAML | Cross-slot donor KV seed |
| `ZEROLLAMA_PREFIX_BLOCK_POOL` | **auto** | On when Radix, LMCache URI, or `n_parallel > 1` |
| `ZEROLLAMA_LMCACHE_URI` | — | Set `file://…` to enable metadata tier |
| `ZEROLLAMA_PREFIX_CACHE_BLOCK_SIZE` | YAML / `512` | EAGLE drop-last-block granularity |
| `ZEROLLAMA_DEBUG` | — | `l3` → prefix trace + decode-graph trace |

Doc: [gpu-profiles-l3.md](./gpu-profiles-l3.md), [radix-prefix-share.md](./radix-prefix-share.md).

---

## Phase 15 KV (in-process)

| Variable | Default | Notes |
|----------|---------|-------|
| `ZEROLLAMA_KV_NATIVE_DECODE` | `1` | C `llama_decode` hot path |
| `ZEROLLAMA_KV_NATIVE_BATCH` | `1` | C batch decode |
| `ZEROLLAMA_KV_NATIVE_SAMPLE` | `1` | C sampling after decode |
| `ZEROLLAMA_KV_AUTO_BATCH` | `0` | Coalesce concurrent `generate()` (adds TTFT wait) |
| `ZEROLLAMA_KV_AUTO_BATCH_MS` | `5` | Auto-batch coalesce window |

Disable native path for A/B: `ZEROLLAMA_KV_NATIVE_DECODE=0`.

---

## VRAM / admission (YAML-first)

Most knobs live under **`vram:`** in runtime YAML (`single_gpu.yaml`, `apple_silicon.yaml`, …). At process start, unset env vars are filled from YAML via `apply_vram_defaults_from_config()`. Effective values also appear on `/health` under `llama_cache.runtime_env.vram` (from `runtime/env.py`).

Common overrides (env wins):

| Variable | YAML key | Role |
|----------|----------|------|
| `ZEROLLAMA_RUNTIME_VRAM_MIN_FREE` | `min_free` | Admission headroom |
| `ZEROLLAMA_RUNTIME_TRAINING_VRAM_RESERVE` | `training_reserve` | Training queue reserve |
| `ZEROLLAMA_RUNTIME_INFERENCE_POLICY` | `inference_policy` | `inference-first` default |
| `ZEROLLAMA_RUNTIME_VRAM_MARGIN` | `margin` | Estimate × margin floor |
| `ZEROLLAMA_RUNTIME_VRAM_PROBE` | (env) | `auto` \| `nvml` \| `smi` |
| `ZEROLLAMA_RUNTIME_VRAM_PROBE_CALIBRATE` | `probe_calibrate` | Post-load observed vs estimate |
| `ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR_AUTOTUNE` | `estimate_factor_autotune` | Per-GGUF calibration |
| `ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX` | `clamp_num_ctx` | Opt-in ctx clamp |
| `ZEROLLAMA_RUNTIME_RAM_OVERHEAD` / `RAM_MARGIN` | — | Host weight / budget multipliers |
| `ZEROLLAMA_RUNTIME_VRAM_WEIGHT_TENSOR` | — | `on` \| `off` \| `auto` (partial `-ngl`) |
| `ZEROLLAMA_RUNTIME_VRAM_KV_BLOCK_LAYOUT` | — | GGUF block layout for KV bytes |
| `ZEROLLAMA_RUNTIME_VRAM_AUTOTUNE_PERSIST` | — | Persist calibrated factors |
| `ZEROLLAMA_RUNTIME_SHARED_PYTHON` | — | Embedded training+runtime interpreter |

Effective values (including YAML-applied knobs) appear on `/health` under `llama_cache.runtime_env.vram`.

Doc: [phase13-runtime-vram.md](./phase13-runtime-vram.md), [scheduling-vram-policy.md](./scheduling-vram-policy.md).

---

## llama.cpp paths

| Variable | Resolution |
|----------|------------|
| `LLAMA_CPP_ROOT` | Bare sibling `../llama.cpp` → **vendor pin** preferred (patched routes) |
| `LLAMA_SERVER_BIN` | Explicit path wins; else under resolved root |

**Patch drift:** `./scripts/vendor/llama_patch_doctor.sh` — patch file count, in-tree `/kv/seq-copy`, vendor commit count, resolved binary path. Included in `./scripts/runtime/runtime_env_doctor.sh` as `llama_patches`.
| `LLAMA_SERVER_BIN` | Explicit override, else root → vendor fallback |
| `LLAMA_CPP_LIB` | Follows same root order |

---

## Anti-patterns

1. **Exporting Metal tuning env then running CUDA smokes** — disk cache and profile ctx leak. Use profiles or `./scripts/runtime/runtime_env_doctor.sh`.
2. **Six L3 env vars when one profile suffices** — use `ZEROLLAMA_L3_PROFILE=agent`.
3. **St stale `LLAMA_CPP_ROOT=../llama.cpp`** — runtime auto-prefers vendor; rebuild vendor if `/kv/seq-copy` 404s.
