# Operations (Phase 7)

**VRAM admission (Phase 11 — why two env knobs, priorities):** [../../docs/phase11-runtime-admission.md](../../docs/phase11-runtime-admission.md). Full Go+Python policy: [../../docs/scheduling-vram-policy.md](../../docs/scheduling-vram-policy.md).

## Option A — two terminals (debug)

```bash
# Terminal 1
export LLAMA_SERVER_BIN=.../llama-server
export LLAMA_MODEL=.../model.gguf
zerollama-runtime serve --port 8081

# Terminal 2
export ZEROLLAMA_RUNTIME_URL=http://127.0.0.1:8081
zerollama serve
```

## Option B — single command (recommended)

From repo root after `pip install -e runtime/.[serve]`:

```bash
export LLAMA_SERVER_BIN=.../llama-server
export LLAMA_MODEL=.../model.gguf
./scripts/serve_with_runtime.sh
# or: zerollama-runtime up
```

This sets `ZEROLLAMA_RUNTIME_URL`, waits for `GET /health`, then runs `zerollama serve` in the foreground. Ctrl+C stops both.

Flags: `zerollama-runtime up --config runtime/configs/dual_4090_ngram.yaml --runtime-port 8081`

**Single GPU (e.g. RTX 5080 16GB):**

```bash
export ZEROLLAMA_RUNTIME_CONFIG=.../runtime/configs/single_gpu.yaml
# Why: dual_4090 tensor split fails on one card; use quantized GGUF for VRAM.
```

## What stays separate

- **Go** (`zerollama`): pull, registry, Eliza cloud, API surface, training embed.
- **Python** (`runtime`): GGUF inference, PagedAttention bookkeeping, llama-server subprocess.

Training remains `training.py` via embedded CPython in the Go daemon.

## Internal GPU hooks (not public operator API yet)

| Endpoint | Effect | Why it exists |
|----------|--------|----------------|
| `POST /internal/training-handoff` | Pause inference; stop `llama-server` | Training OOM path and manual “free GPU” before legacy models |
| `POST /internal/inference/resume` | Set state back to `running` | After handoff, generate would otherwise 503 until daemon restart |
| `POST /internal/vram-estimate` | `{"gguf","num_ctx?","options?"}` → `vram_estimate` + `vram_budget` (`host_ram` uses `ZEROLLAMA_RUNTIME_RAM_MARGIN`; present even when GPU free VRAM is unknown) | Loopback-only; same estimate path as `/health` (draft model, `-c` from llama args, parallel slots). CLI: `scripts/runtime_vram_estimate.sh <gguf> [--num-ctx N]`. |
| `POST /v1/chat/completions` | OpenAI-shaped body; Go proxy on `:8080` injects `options.gguf` from manifest. Direct `:8081` may pass `options` / `num_ctx`. Uses `prepare_v1_chat()` / Phase 13 clamp. Non-stream may include `vram_num_ctx`. |

Runtime exposes loopback-only `POST /internal/tokenize` (`gguf`, `text`) for libllama vocab-only tokenize (Phase 14). Go `/internal/render-chat` uses it when no ggml runner is loaded so `truncate_mode` can be `tokenize` on the runtime inference path.

Runtime tools chat calls Go (loopback-only) on `ZEROLLAMA_GO_URL` (auto-set at serve from `OLLAMA_HOST` when bind is `0.0.0.0`; override explicitly if needed): `POST /internal/render-chat` (prompt + `tool_tag` + `truncate_mode` + `truncated`), `POST /internal/parse-tool-output` (one-shot), and streaming `POST /internal/parse-tool-output/session` + `chunk` + `close` (stateful `model/parsers/*` and template parser — same as ggml). `ZEROLLAMA_RUNTIME_GO_RENDER_CHAT=auto` (disable with `off`). Render accepts `num_ctx` and optional `num_predict`; truncation uses ggml **tokenize** when a runner for the model is already loaded, else **heuristic** (~len/4 with completion reserve). Response **`truncated`** is true only when older messages were dropped; **`truncate_mode`** is `none` | `heuristic` | `tokenize`.

Go training OOM also calls handoff via [`internal/runtimeclient`](../../internal/runtimeclient/client.go). **Why inference-first policy still uses handoff:** when training needs VRAM, something must unload the runtime subprocess—default product mode is chat/generate, but training cannot share a 16GB card with a loaded GGUF.

Optional **`ZEROLLAMA_TRAINING_WAIT_INFERENCE_IDLE=1`**: reject new training jobs (`POST /api/train/jobs` and TCP `:9500` `submit_job`) while the ggml scheduler or Python runtime (`GET /health` `waiting`/`running`/`llama_server`) has active inference work. With idle-wait on, a loaded runtime GGUF blocks training until unload — on subprocess backends see `llama_server: true`; on Phase 14 in-process/wheel backends see `inference_state: running` and `llama_model` set (`llama_server` stays `false`). Typical on single-GPU hosts sharing VRAM with training.

Bind the runtime to loopback in production (`ZEROLLAMA_RUNTIME_HOST=127.0.0.1`); `/internal/*` also rejects non-loopback TCP clients as defense in depth.

When `OLLAMA_TRAINING` and `ZEROLLAMA_BLOCK_INFERENCE_DURING_TRAINING` are on (default), the Go daemon also runs a background policy loop: pause ggml loads, call `training-handoff`, and unload all runners while training is active, a job is running, or a training HF model remains on CUDA; it resumes runtime inference when that clears.

### `GET /health` fields (runtime)

| Field | Shape | Meaning |
|-------|--------|---------|
| `gpu_memory` | `[{ "device": 0, "free": "12.3 GiB", "probe": "nvml" }, ...]` | Per-GPU free VRAM for devices in the active config. `probe: host-unified` (Linux iGPU) or **`metal-unified`** (macOS vm_stat). |
| `vram_probe_mode` | string | Configured probe (`auto`, `nvml`, `smi`; shared embed may remap `auto` → `smi`). |
| `vram_probe_effective` | string | Operational probe: `nvml`, `nvidia-smi`, `host-unified`, **`metal-unified`**, or **`skipped`** (shared embed, no `nvidia-smi`). |
| `shared_interpreter` | bool | `true` when training + runtime share one embedded CPython. |
| VRAM pre-check env | — | `VRAM_ESTIMATE_FACTOR`, `VRAM_ESTIMATE_FACTOR_AUTOTUNE=auto`, `VRAM_AUTOTUNE_PERSIST=auto`, `VRAM_ESTIMATE_FACTOR_EXPORT=auto`, `VRAM_APPLY_EXPORTED_ENV=1` (startup: load `vram_estimate_factor.env` when `VRAM_ESTIMATE_FACTOR` unset), `VRAM_APPLY_EXPORTED_ENV_PATH` (override file path), `STATE_DIR` (default `~/.cache/zerollama`), `VRAM_PROBE_CALIBRATE=auto`, `VRAM_CLAMP_NUM_CTX=0` (default off; `1` or `auto` with `CHECK_GPU_VRAM=1` lowers request ctx to suggestion at enqueue), `VRAM_SUGGEST_CTX_MAX`, `VRAM_WEIGHT_*`, `VRAM_KV_BLOCK_LAYOUT=1` (IQ/TQ KV types use ggml block sizes; legacy Q4/Q8 stay ≥2 bytes/elem), `VRAM_KV_*`, `CHECK_GPU_VRAM`. **Autotune:** per-model factors in `vram_autotune.json`; **export:** `vram_estimate_factor.env` (last load) and `vram_autotune_factors.env` (catalog). |
| `vram_autotune` | object | `enabled`, factors, `persist` (incl. `catalog[]`, `catalog_truncated`), `export`, `apply_exported_env` (startup apply status). Autotune persist wins per GGUF over applied env. `pending_first_calibration`. |
| cross-queue | `admission.cross_queue_pressure` | Dashboard sum (runtime + defer + ggml backlog). Admission uses the existing defer/ggml/runtime gates — not this scalar. |
| `vram_estimate` | object | Loaded model or `LLAMA_MODEL` when set; uses `loaded_vram_num_ctx` when llama-server is up. Includes `estimate_factor_source` (`env` / `session` / `catalog`), `kv_block_layout`, `kv_bytes_*`, `head_dim_source`; IQ/TQ KV types floor at F16 bytes/element. |
| `loaded_vram_num_ctx` | int \| null | Context length used for the loaded llama-server (for estimate parity). |
| `vram_budget` | object | `model_gguf`, `required_per_gpu` vs `free_bottleneck`, `fits` / `fits_with_margin` (uses `VRAM_MARGIN`), `admission_*` when VRAM gate on. `admission` also mirrors `vram_load_fits`, `vram_admission_fits`. |
| `vram_calibration` | object | Last load: `observed_bytes`, `estimated_raw_bytes`, `estimated_effective_bytes` (= raw × `suggested_estimate_factor` when probe succeeded), optional `estimated_precheck_bytes` (factor used at pre-check). `suggested_estimate_factor` = observed/raw — **set** `VRAM_ESTIMATE_FACTOR` to that value (replace, do not multiply). Clamped 0.1–3. `autotune_active` / `probe_calibrate_required_for_autotune` on snapshot. Informational only. |
| `autoconfig` | object | `config_path`, `pick` (`single_gpu` / `dual_4090` / `apple_silicon` / `custom`), `visible_gpu_count`, optional `gpu_total_vram`. |
| `gpu_profile` | object | **L1 autotune** when enabled: `id`, `name`, `source` (`match` / `bucket` / `apple_memory`), `bucket_label`, `n_parallel`, `ctx_size_default`, `unified_memory_gb` (darwin), `cache_types_fallback`, `emit_ctx_size`, `emit_mlock`. Absent when `ZEROLLAMA_GPU_PROFILE=0`. Doc: [../../docs/gpu-profiles-l1.md](../../docs/gpu-profiles-l1.md). |
| `llama_cache` | object | **L3 prefix cache** when enabled: `enabled`, `root`, `default_ttl_ms`, `model_path`, `model_loaded`, `model_hash`, `slot_save_path`, `file_count`, `files[]`. **Why:** operators verify session KV disk cache without reading slot dirs. Absent fields shrink when `ZEROLLAMA_LLAMA_CACHE=0`. Doc: [../../docs/gpu-profiles-l3.md](../../docs/gpu-profiles-l3.md). |
| `llama_backend` | string | Effective forward backend: `subprocess`, `inprocess`, or `llama-cpp-python`. |
| `llama_backend_source` | string | `env` when `ZEROLLAMA_RUNTIME_LLAMA_BACKEND` is set; `config` when the loaded YAML has an explicit `llama_backend` key; else `default` (packaged subprocess). |
| `llama_cpp` | object | **Wheel backend only** (key absent otherwise): `gpu_mode` (`cpu`/`gpu`), `n_gpu_layers`, `loaded`, optional `env_n_gpu_layers`. |
| `vram_budget.suggested_max_num_ctx` | int | Largest `num_ctx` where `max(estimate×margin, admission_min_free)` fits `admission_effective_free_bytes` (falls back to `free_bottleneck` when admission fields absent). `num_ctx_over_budget` when current ctx exceeds it. VRAM **reject** errors append `try num_ctx<=N`. Cap: `VRAM_SUGGEST_CTX_MAX` (default 131072). |
| `vram_num_ctx_policy` | object | `clamp_enabled` — when true (`VRAM_CLAMP_NUM_CTX=1` or `auto` with `CHECK_GPU_VRAM=1`), request `num_ctx` above suggestion is lowered at enqueue. Responses/stream first chunk include `vram_num_ctx` when clamped (`num_ctx_clamped`, `num_ctx_clamped_from`, effective `num_ctx`). |
| `host_memory` | `{ "available", "swap_free", "load_budget" }` | Linux `/proc/meminfo` budget for GGUF host pre-check. |
| `admission` | object | Queue cap; VRAM gate; training reserve; **`go_training_gpu_busy`**; **`options.priority`**; **`gates_active`** — when true, only **`priority: low`** is throttled (`low_would_wait`, `defer_would_block_low`, …). **`runtime_backlog_pressure`** does not block normal chat. Legacy keys in **`gates_active_compat`**. |
| `go_coordination` | object | Go mirror: `defer_waiting`, `sched_pending` / `sched_active` / `sched_loaded`, `training_gpu_blocked`, `ggml_loads_paused`, plus `coordination.{fresh,stale,age_s,ttl_s}`. Stale mirrors (> `GO_COORDINATION_TTL_S`, default 30s) are ignored (fail-open). |
| inference-first | `admission.inference_policy` + `admission.backpressure` | **On by default** (fixed thresholds in code: runtime backlog ≥4, defer ≥1, ggml sched ≥1, cross-queue pressure ≥6). Only `priority: low` / batch throttled at **enqueue and dequeue**; `normal` chat is not stalled by those mirrors; `high` bypasses min-free gate and jumps the queue. Disable with `ZEROLLAMA_RUNTIME_INFERENCE_POLICY=off` (VRAM checks stay on unless `CHECK_GPU_VRAM=0`). |
| VRAM headroom | `admission.vram_min_free_configured`, `admission.vram_training_reserve_configured` | Defaults **1 GiB** / **2 GiB**; override with `ZEROLLAMA_RUNTIME_VRAM_MIN_FREE`, `ZEROLLAMA_RUNTIME_TRAINING_VRAM_RESERVE` (size strings — prefer **GiB**; `GB`/`MB` use decimal 1000-based units). |
| enqueue / dequeue VRAM | — | When `ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM=1` (default), host + GGUF GPU budget runs **before queueing** (when path known) and again at dequeue if VRAM changed. **2 GiB** training reserve (code constant) while handoff / pause / `go_training_gpu_busy`. Fail closed if GPU probe unavailable. |

Policy across Go + Python (VRAM broker, T6 defer queue, why not one FIFO): [../../docs/scheduling-vram-policy.md](../../docs/scheduling-vram-policy.md). Smoke checklist: [../../docs/testing-smoke.md](../../docs/testing-smoke.md).

## Streaming

`/api/generate` and `/api/chat` support **streaming** (`stream: true` or omitted, Ollama default) via the runtime when proxied. The runtime translates llama-server SSE into Ollama NDJSON. **Tools:** Go `/internal/render-chat` + stateful `/internal/parse-tool-output/*` for builtin family parsers; generic JSON prompt/parser when Go is off or the model has no `parser` in Modelfile. Vision and logprobs still use the legacy runner.

## OpenAI API

`POST /v1/chat/completions` (text + tools, no vision/logprobs) is proxied to the runtime when `ZEROLLAMA_RUNTIME_URL` is set. Streaming uses OpenAI SSE (`text/event-stream`).

## E2E smoke

```bash
# runtime must be listening on 8081
./scripts/e2e_runtime_smoke.sh

# with GPU + model (env must be set on the *serve* process)
RUN_E2E_GPU=1 LLAMA_MODEL=/path/to/model.gguf LLAMA_SERVER_BIN=.../llama-server ./scripts/e2e_runtime_smoke.sh

# through zerollama proxy — needs zerollama serve; uses X-Zerollama-Runtime: 1
export OLLAMA_HOST=http://127.0.0.1:8080
RUN_E2E_PROXY=1 ./scripts/e2e_runtime_smoke.sh   # generate/chat/v1 + stream via Go proxy
RUN_E2E_GPU=1 RUN_E2E_TOOLS=1 ./scripts/e2e_runtime_smoke.sh   # optional tools on :8081 (+ proxy when RUN_E2E_PROXY=1)
./scripts/gpu_smoke_all.sh                       # coordination + full GPU/proxy smokes
RUN_E2E_TOOLS=1 ./scripts/gpu_smoke_all.sh         # full smoke + tools path
./scripts/gpu_health_report.sh                   # /health tuning summary after a probed load
```

**Why `smoke` model name in scripts:** placeholder for ad-hoc proxy tests (`X-Zerollama-Runtime: 1`); use `RUN_E2E_GGUF` or `LLAMA_MODEL` on serve. Pulled tags use manifest `options.gguf` via Go proxy (Phase 9).

**Health report module:** `runtime.gpu_health_report.format_gpu_health_tuning_report` — `python -m runtime.gpu_health_report` with `HEALTH_JSON` set.

## Build llama-server

See [`../../scripts/build_llama_server.sh`](../../scripts/build_llama_server.sh). **Why arch matters:** RTX 5080 needs `CMAKE_CUDA_ARCHITECTURES=120-real` and a toolkit whose `nvcc` exists (see smoke doc).

Optional: `export LLAMA_SERVER_EXTRA_ARGS="-c 8192"` to cap context on tight VRAM.
