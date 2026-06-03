# Phase 13 — Runtime VRAM estimates, context policy, and single-GPU autoconfig

**Audience:** Operators on tight GPUs (e.g. RTX 5080 16 GB) and contributors working on Python pre-checks, `/health`, and load admission.

**Related:** [scheduling-vram-policy.md](./scheduling-vram-policy.md) (full Go+Python stack), [phase11-runtime-admission.md](./phase11-runtime-admission.md) (who gets the GPU when busy — **not** estimate tuning), [runtime/docs/OPERATIONS.md](../runtime/docs/OPERATIONS.md) (field reference), [testing-smoke.md](./testing-smoke.md) (GPU smokes).

---

## Why Phase 13 exists

Phase 11 answers **“should we admit this request?”** when the GPU is busy or nearly full. Phase 13 answers **“how much VRAM will this GGUF need at this `num_ctx`?”** before starting `llama-server`.

**Why separate from Phase 11:** admission thresholds (defer, ggml backlog, priority) are product policy; **estimate calibration** is per-model physics (quant, layers, KV, draft). Mixing dozens of `ADMISSION_*` env vars with estimate knobs made operators tune the wrong layer.

**Why heuristics instead of only NVML after load:** subprocess OOM on a full 16 GB card is slow, opaque, and hard to recover from. Pre-check + suggest + optional clamp turn failures into **actionable HTTP errors** (`try num_ctx<=N`) and dashboard fields.

---

## Operator surface (estimate + context)

| Variable | Default | Why |
|----------|---------|-----|
| `ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM` | on when probe works | Master switch for host + GPU budget checks (Phase 11 uses same gate for 1 GiB floor). |
| `ZEROLLAMA_RUNTIME_VRAM_MARGIN` | `1.0` | Multiplier on estimated bytes at pre-check (safety margin). |
| `ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR` | `1.0` | Global scale on final estimate; replace with `vram_calibration.suggested_estimate_factor` after probe (do not multiply on top of autotune). |
| `ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR_AUTOTUNE` | `auto` | Apply per-GGUF factor from last calibration / `vram_autotune.json`. |
| `ZEROLLAMA_RUNTIME_VRAM_PROBE_CALIBRATE` | `auto` | Record NVML/smi free VRAM before/after load → `/health` `vram_calibration`. |
| `ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX` | **`0` (off)** | When `1` or `auto` (with GPU checks on), lower request `num_ctx` to `suggested_max_num_ctx` at enqueue. **Why default off:** silent context reduction surprised operators; use `auto` on known-tight single-GPU smoke only. |
| `ZEROLLAMA_RUNTIME_VRAM_SUGGEST_CTX_MAX` | `131072` | Upper bound for binary search in `suggested_max_num_ctx`. |
| `ZEROLLAMA_RUNTIME_VRAM_MIN_FREE` | `1GiB` | Phase 11 admission floor (prefer **GiB**; `GB` uses decimal 1000-based bytes). |
| `ZEROLLAMA_RUNTIME_TRAINING_VRAM_RESERVE` | `2GiB` | Headroom while training / handoff / Go `training-gpu-busy`. |
| `ZEROLLAMA_AUTO_CONFIG` | `1` | Pick `configs/single_gpu.yaml` vs `dual_4090.yaml` from visible GPU count. |
| `ZEROLLAMA_RUNTIME_CONFIG` | autoconfig | Explicit YAML path overrides autoconfig. |

**Apply exported factors at startup:** `ZEROLLAMA_RUNTIME_VRAM_APPLY_EXPORTED_ENV=1` loads `vram_estimate_factor.env` when `VRAM_ESTIMATE_FACTOR` is unset. **Why:** systemd units can `source` one file without hand-editing unit env; per-GGUF autotune persist still wins at load time.

---

## Data flow

```text
Request options.num_ctx / -c / GGUF metadata
        │
        ▼
resolve_num_ctx_for_request()  ──► optional VRAM clamp (if VRAM_CLAMP enabled)
        │                              same path for tools render + _admit_one
        ▼
enqueue: host RAM + check_gguf_vram_budget
        │
        ▼
dequeue: re-check (VRAM may have dropped)
        │
        ▼
llama-server load (-c matches admitted num_ctx)
        │
        ▼
VRAM_PROBE_CALIBRATE (optional) → vram_calibration + autotune persist
```

**Why `resolve_num_ctx_for_request` is shared:** tools chat used to render prompts with **uncapped** `num_ctx` while load used **capped** context — truncation and KV sizing disagreed. One function keeps render, precheck, queue, and `-c` aligned.

---

## `/health` and APIs

| Surface | Role |
|---------|------|
| `vram_estimate` | Bytes breakdown for loaded or probed GGUF (`kv_cache_bytes`, `estimate_factor_effective`, `estimate_factor_source`: `env` / `session` / `catalog`, …). |
| `vram_budget` | `fits`, `fits_with_margin`, `suggested_max_num_ctx`, `num_ctx_over_budget`. |
| `vram_calibration` | Last load: `suggested_estimate_factor` ≈ observed/raw. |
| `vram_autotune` | Per-model persist status; `persist.catalog[]` lists calibrated GGUFs (`model`, `basename`, `estimate_factor`, `last`). |
| `autoconfig` | Which YAML was chosen (`single_gpu` / `dual_4090` / `custom`). |
| `vram_num_ctx_policy` | Whether clamp is enabled and env value. |
| `POST /internal/vram-estimate` | Loopback-only; same math as `/health` without load. |

**Client-visible clamp:** when clamp runs, Ollama `/api/chat` and `/api/generate` include top-level `vram_num_ctx`. OpenAI `/v1/chat/completions` non-stream responses include the same field (extension). **Why:** operators and agents can see context was lowered without reading logs only.

**v1 `options`:** Go `runtimeV1ChatCompletionsProxy` injects `options.gguf` from the manifest (same as `/api/chat`). Direct `:8081` callers may also pass `options.gguf`, `options.num_ctx`, or top-level `num_ctx`. `prepare_v1_chat()` uses the same `resolve_num_ctx_for_request` path as `/api/chat`.

---

## Operator CLI and smokes

```bash
# Pre-flight budget (no load)
./scripts/runtime_vram_estimate.sh /path/to/model.gguf --num-ctx 8192

# Coordination + Phase 13 health fields
ZEROLLAMA_RUNTIME_URL=http://127.0.0.1:8081 ./scripts/e2e_coordination_smoke.sh

# GPU: estimate → generate → calibration hint
RUN_E2E_GPU=1 LLAMA_MODEL=... LLAMA_SERVER_BIN=... ./scripts/e2e_runtime_smoke.sh

# Optional: tools + proxy + clamp probe (serve must enable clamp for last)
RUN_E2E_GPU=1 RUN_E2E_PROXY=1 RUN_E2E_TOOLS=1 ./scripts/e2e_runtime_smoke.sh
RUN_E2E_VRAM_CLAMP=1 ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX=auto ./scripts/e2e_runtime_smoke.sh
```

**Why `runtime_vram_estimate.sh`:** pick quant and `num_ctx` before paying load latency; same code path as Go `LogVramBudgetIfTight` async probe on proxy.

---

## Code map

| Layer | Path |
|--------|------|
| Suggest + clamp | `runtime/runtime/vram_suggest.py` |
| Estimate + budget | `runtime/runtime/gpu_vram.py`, `gguf_estimate.py` |
| Admit + load ctx | `runtime/runtime/engine.py` (`resolve_num_ctx_for_request`, `_admit_one`) |
| HTTP | `runtime/runtime/server/app.py` (`_request_num_ctx`) |
| Tools stream meta | `runtime/runtime/server/chat_tools.py` |
| Autoconfig | `runtime/runtime/autoconfig.py` |
| YAML `vram:` defaults | `runtime/runtime/vram_yaml_defaults.py`, `configs/single_gpu.yaml` |
| Snapshot recommend | `runtime/runtime/gpu_snapshot.py` (`python -m runtime.gpu_snapshot`) |
| Calibration / autotune persist | `runtime/runtime/vram_calibration.py`, `vram_autotune_persist.py` |
| Go async probe | `internal/runtimeclient/vram_estimate.go` |
| Tests | `test_vram_suggest.py`, `test_gpu_vram.py`, `test_internal_vram_estimate.py`, `test_vram_autotune*.py` |

---

## Known limits

| Limit | Why |
|-------|-----|
| Estimates are heuristic | Exact KV when metadata complete; else conservative floors. Use probe calibrate + autotune on real weights. |
| Binary search assumes monotone VRAM vs `num_ctx` | True for KV scaling; rare odd manifests may mis-suggest. |
| Render truncation (Phase 12) | Go `/internal/render-chat` is heuristic (`num_predict` reserve when set); not tokenizer-exact — separate polish item. |
| Phase 14 in-process | Subprocess remains default; ctypes `inprocess` improves KV accounting (Phase 15) when enabled via env or YAML. |

---

## Why `vram:` in `single_gpu.yaml`

Autoconfig already selects this file on one visible GPU. The `vram:` block is applied at process start by `vram_yaml_defaults.py` **only for env vars that are unset** — **why:** ship sensible 16GB defaults in-repo without forcing every operator to duplicate `1GiB` / `2GiB` / autotune in systemd; production overrides stay in env. Clamp defaults to **`"0"` (off)** — **why:** silent `num_ctx` reduction was surprising; enable `auto` deliberately.

Full narrative: [gpu-5080-operator-guide.md](./gpu-5080-operator-guide.md).

---

## Suggested 5080 workflow

1. `export ZEROLLAMA_RUNTIME_CONFIG=.../runtime/configs/single_gpu.yaml` (or rely on `ZEROLLAMA_AUTO_CONFIG=1`)
2. `./scripts/runtime_vram_estimate.sh "$LLAMA_MODEL" --num-ctx 8192` — read `suggested_max_num_ctx` and `fits_with_margin`.
3. One probed load with `VRAM_PROBE_CALIBRATE=auto` — read `/health` `vram_calibration.suggested_estimate_factor` or rely on autotune persist.
4. `./scripts/gpu_health_report.sh` (or `python -m runtime.gpu_health_report` with `HEALTH_JSON`) — post-load summary + suggested env exports.
5. Tune `VRAM_MIN_FREE` / `TRAINING_VRAM_RESERVE` from measured free VRAM under chat+training (see `/health` `vram_*_configured`).
6. Enable `VRAM_CLAMP_NUM_CTX=auto` only if you accept automatic context lowering (watch for `vram_num_ctx` in API responses).
7. `./scripts/gpu_phase13_snapshot.sh --gguf "$LLAMA_MODEL" --num-ctx 8192` — save JSON (`GPU_PHASE13_SNAPSHOT_OUT=5080.json`); prints `python -m runtime.gpu_snapshot` hints (`GPU_SNAPSHOT_RECOMMEND=0` to skip).
8. `./scripts/gpu_5080_session.sh` — preflight + smokes + snapshot + recommendations (official 16GB gate).
9. `./scripts/gpu_smoke_all.sh` — coordination + full inference smokes after rebuild.
10. Optional: `RUN_E2E_TOOLS=1 ./scripts/gpu_smoke_all.sh` — `/api/chat` with tools on `:8081` and `:8080` proxy (not legacy 501).
11. Optional: `RUN_E2E_VRAM_CLAMP=1` with `ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX=auto` on serve — smoke asserts `vram_num_ctx_policy.clamp_enabled` and probes high `num_ctx`.
