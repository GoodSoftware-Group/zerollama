# Scheduling, VRAM, and queue policy

This document explains **how inference and training share one machine** in zerollama today: what is implemented, what is intentionally *not* unified yet, and **why** each knob exists.

It complements:

- [gpu-training.md](./gpu-training.md) — embedded Python training, OOM bridge, `/api/train/*`
- [runtime/docs/OPERATIONS.md](../runtime/docs/OPERATIONS.md) — Python runtime ops, `/health`, internal handoff
- [testing-smoke.md](./testing-smoke.md) — dual-stack smoke on a single GPU
- [ROADMAP.md](./ROADMAP.md) — phases 8–13 and training track T6
- [fleet-scheduling.md](./fleet-scheduling.md) — multi-node management, warm routing, agent status (fleet track F1–F7)
- [localai-borrowings.md](./localai-borrowings.md) — fast GGUF metadata, guess hooks, watchdog, concurrency groups, manifest hygiene
- [upstream-ollama-diff.md](./upstream-ollama-diff.md) — upstream uses Go→llama-server only (no Python runtime); Phase 17 alignment

---

## Why this is not one queue yet

**Inference** (chat/generate → Go ggml scheduler and/or Python `llama-server`) and **training** (`/api/train/*` → embedded `training.py`) are separate FIFOs on purpose:

| Queue | Owner | Why separate |
|-------|--------|----------------|
| ggml runners | Go `server/sched.go` | Mature runner lifecycle, tools/vision, keep-alive |
| Runtime requests | Python `runtime/` scheduler | PagedAttention bookkeeping, subprocess `llama-server` |
| Training jobs | Python `training.py` job thread | PyTorch `Trainer` loops are not safe to preempt mid-epoch |

A single global optimizer (“drain chat, then train all night”) is **roadmap** ([T6](./ROADMAP.md#phases-training-track)). Today we **coordinate** through:

1. **VRAM broker** (Go) — proactive unload before loads
2. **Training GPU monitor** — block/evict inference while training holds the card
3. **T6 idle-wait + defer queue** — optional “do not start training until inference is quiet” ([operator guide](./t6-unified-queue.md))

**Why document this:** operators on a 16 GB card often expect one knob (`OLLAMA_MAX_LOADED_MODELS=1`) to fix everything; these layers address different failure modes (OOM vs 409 vs silent defer).

---

## Architecture (one GPU)

```text
Clients
  ├─ /api/chat, /api/generate  ──► Go proxy? ──► Python runtime (llama-server)
  │                              └─► ggml runner (tools, vision, legacy)
  └─ /api/train/jobs           ──► Go policy ──► embedded training.py

Shared resources
  ├─ VRAM broker (server/vram)     training-handoff, UnloadAllRunners, PrepareForTraining
  ├─ Training policy monitor     pause ggml loads while training busy
  └─ T6 workload gate            optional reject/defer training submit
```

**Why Go still owns policy:** Python cannot see ggml runner processes or the public HTTP surface; Go cannot run arbitrary `Trainer` graphs without embedded CPython.

---

## Phase 8 — VRAM broker (Go)

**What:** Before heavy GPU use, Go calls into the runtime internal API and/or evicts ggml runners.

| Trigger | Action | Why |
|---------|--------|-----|
| Legacy ggml load (runtime model) | `training-handoff` | Runtime holds `llama-server`; must free VRAM first |
| Runtime proxy inference | `UnloadAllRunners` (+ pins kept) + resume only if ggml clear; same-GGUF+empty skip; pin GGUF mismatch / residual ggml → 503; exclusive bench → `Forced` | Avoid OOM and lying multi-GGUF warm |
| Training / runtime broker | `UnloadAllRunnersForced`, `training-handoff` | Free VRAM before another stack uses the GPU (pins ignored) |
| Training submit / OOM | `PrepareForTraining` | Same card; training needs a predictable empty budget |

**Why no public `/api/runtime/unload`:** automatic eviction avoids operators forgetting resume and leaving inference 503 until restart ([ROADMAP](./ROADMAP.md) deprioritizes public unload).

Code: `server/vram/broker.go`, wired from `server/routes.go` and `x/trainingworker`.

### Phase B addenda (pin / thrash dampen)

**Why this belongs in the VRAM policy doc:** Phase B is not a new scheduler — it changes *when* the Phase 8 broker unloads and *who* unload may keep.

| Mechanism | Behavior | Why |
|-----------|----------|-----|
| Soft `UnloadAllRunners` | Keeps pin + fulfillment-protected ggml keys | Otherwise `/api/pin` is a no-op against every runtime chat |
| `UnloadAllRunnersForced` | Clears protected keys too | Training OOM and exclusive bench must reclaim the card |
| B0 skip unload | Same GGUF warm **and** ggml map empty | Match alone left ggml resident → dual-stack OOM |
| Pin residual / GGUF conflict | `503` before `ResumeInference` | Fail closed; do not resume Python on top of pinned ggml |
| Warm hysteresis | Prefer cold idle victims for `ZEROLLAMA_WARM_HYSTERESIS` | Reduce ggml ping-pong among recently used runners |

Full client contract: [inference-wishlist-host.md](./inference-wishlist-host.md).

---

## Go ggml scheduler: `keep_alive`, unload, and `num_ctx` at load

**Why this section exists:** operators use `/api/create`, `zerollama stop`, and `/api/ps` together; behavior differs from the Python runtime path (Phase 13 per-request `-c`). Misunderstanding load-time KV sizing caused “manifest says 262K, `/api/ps` says 4096, stop did nothing, create hung.”

### Unload (no public `/api/unload`)

| Mechanism | API | Why |
|-----------|-----|-----|
| `zerollama stop MODEL` | `POST /api/generate` with `"prompt":""`, `"keep_alive":0` | Same contract as upstream Ollama; avoids a second unload API |
| Chat unload | `POST /api/chat` with `"messages":[]`, `"keep_alive":0` | Symmetric with generate |
| Post-create eviction | Successful `/api/create` → `expireRunner` | Manifest options changed; warm runner must not keep old `num_ctx` |
| Training / runtime broker | `UnloadAllRunnersForced`, `training-handoff` | Free VRAM before another stack uses the GPU |

**Verify:** `GET /api/ps` should list no models after unload. If a model remains, an inference request may still hold a ref (`refCount > 0`); unload retries until refs drop.

Code: `server/sched.go` (`expireRunner`, `findLoadedRunner`, `processExpiredRunner`), `server/routes.go` (generate/chat early return), `cmd/cmd.go` (`StopHandler`).

### `num_ctx`: manifest default vs request options

**Merge order** (`server/routes.go` → `modelOptions`): VRAM-tier server default → **manifest `parameters`** → **request `options`**.

**At load** (`sched.go` → `load` → `llama.Load`): runner options (`NumCtx`, `NumGPU`, `NumBatch`, …) fix KV size. **`needsReload`** compares stored runner options to the merged options on the next request — a manifest or request change to `num_ctx` forces evict + reload with the new size.

| Set where | Typical use | Risk |
|-----------|-------------|------|
| Manifest `parameters.num_ctx` | Default for all loads | **Pre-allocates full KV at load** — very large values (262144) can hang on qwen35/recurrent models |
| Request `options.num_ctx` | Hermes, long single-shot prompts | May reload runner if ≠ loaded; still allocates KV for requested size at load of that runner |
| `/api/ps` `context_length` | Observability | **Loaded** runner only — not manifest, not per-request unless reload occurred |

**Guidance:** keep manifest defaults modest (4096–8192 on Mac ggml); pass large context via **`options.num_ctx`** when needed. Python runtime path uses `resolve_num_ctx_for_request` — see [phase13-runtime-vram.md](./phase13-runtime-vram.md).

### Ggml VRAM suggest and opt-in clamp (M12, Jun 2026)

**Why:** Phase 13 runtime answers “what is the largest `num_ctx` that fits?” before enqueue. The Go ggml scheduler merges manifest defaults at load time with no equivalent signal — operators on Metal/legacy saw hangs with `num_ctx: 262144` (high-VRAM tier default + manifest) and no API hint. Clamp defaults **off** on both paths so context is never reduced silently unless opted in.

**When it runs:** `scheduleRunner` → `applyGgmlNumCtxClamp` after `modelOptions`, before `GetRunner`. `/api/show` calls suggest only (no clamp).

**Estimate path:** `llm.LoadModel` → `fs/ggml.GraphSize` (KV + graph scratch) + on-disk file size (weights proxy) × `ZEROLLAMA_GGML_VRAM_MARGIN` (default 1.05) vs **free** VRAM after `GpuOverhead()` + `MinimumMemory()` per device — same floor as load logging in `sched.go`. Binary search 512 … `min(train_ctx, ZEROLLAMA_GGML_SUGGEST_CTX_MAX)`.

**Free VRAM discovery:**

| Path | Behavior | Why |
|------|----------|-----|
| Load (chat/generate) | `GPUDevices(ctx, LoadedRunnersForDiscovery())` — refresh | Layer fit must see current headroom; loaded runners report accurate free bytes |
| `/api/show` | 2s TTL cache on server | CLI and UIs call show often; full probe per request was too slow |
| Free unknown | Suggest omitted (no fields) | **Why:** total installed VRAM ≠ free; an early fallback to startup total over-suggested context and could still OOM |

**API (`ggml_num_ctx` object):**

| Field | When set | Meaning |
|-------|----------|---------|
| `suggested_max_num_ctx` | Show + load (when computable) | Largest ctx estimate that fits free VRAM |
| `merged_num_ctx` | Show only, when merged default > suggestion | Merged server/manifest default that exceeds VRAM — **not** clamped |
| `num_ctx_clamped` | Load response when clamp env on | True when context was lowered before load |
| `num_ctx_clamped_from` | With clamp | Original merged/request ctx |
| `num_ctx` | With clamp | Effective ctx after clamp (load responses only) |

Example show when manifest tier default is too high:

```json
"ggml_num_ctx": {
  "suggested_max_num_ctx": 8192,
  "merged_num_ctx": 262144
}
```

**Env (default off — parity with Phase 13):**

| Variable | Default | Why |
|----------|---------|-----|
| `ZEROLLAMA_GGML_CLAMP_NUM_CTX` | `0` | Lower merged `num_ctx` to suggestion before load; operators expect requested context unless opted in |
| `ZEROLLAMA_GGML_SUGGEST_CTX_MAX` | `131072` | Upper bound for binary search (matches runtime `VRAM_SUGGEST_CTX_MAX`) |
| `ZEROLLAMA_GGML_VRAM_MARGIN` | `1.05` | Conservative multiplier on estimate vs free bytes |
| `ZEROLLAMA_GGML_AUTO_PARALLEL` | `auto` (on) | Fit llama-server `-np` from free VRAM at load; Go completion semaphore matches `-np` |
| `ZEROLLAMA_GGML_PARALLEL_MAX` | `8` | Upper bound for auto `-np` when `OLLAMA_NUM_PARALLEL` is unset |
| `OLLAMA_NUM_PARALLEL` | `1` (upstream default) | Hard **cap** when auto is on (explicit env); exact value when auto is off |

Code: `server/ggml_num_ctx.go`, `envconfig/ggml_num_ctx.go`, `server/sched.go` (`LoadedRunnersForDiscovery`).

### Scheduler watchdog and concurrency groups (Jun 2026)

**Why:** `OLLAMA_MAX_LOADED_MODELS` caps resident runners but does not reclaim under VRAM pressure, recover stuck loads, or prevent **incompatible pairs** (chat + imagegen on 16 GB). LocalAI’s WatchDog pattern maps to a lightweight Go goroutine—not a second scheduler.

| Mechanism | Env / manifest | Why |
|-----------|----------------|-----|
| Idle LRU + VRAM reclaim | `ZEROLLAMA_MEMORY_RECLAIM_THRESHOLD` (e.g. `0.95`) | Evict least-recently-used idle runner when GPU usage crosses threshold |
| Operator VRAM ceiling | `ZEROLLAMA_VRAM_BUDGET` (`80%` or `12GiB`) | Scheduler and runtime probes see `min(detected, budget)` so loads cannot use reserved headroom |
| Busy timeout | `ZEROLLAMA_RUNNER_BUSY_TIMEOUT` (e.g. `30m`) | Unload runners stuck in-flight (agent disconnect, bad client) |
| Tick | `ZEROLLAMA_SCHED_WATCHDOG_INTERVAL` (default `30s`) | Periodic reclaim/busy checks |
| Concurrency groups | `PARAMETER concurrency_groups ["vram-heavy"]` | Evict conflicting residents **before** loading a model in the same group |
| Load coalescing | (built-in) | Concurrent pulls/loads for same model share one load |
| Pull dedup | `singleflight` on `PullModel` | Avoid duplicate registry work |

**Training idle-wait:** `InferenceBacklog().loaded` counts **all** runners in the scheduler map (including loading). Fleet `inference.ggml.loaded` counts **ready** runners only — see [fleet-scheduling.md](./fleet-scheduling.md).

Full reference: [localai-borrowings.md](./localai-borrowings.md). Code: `server/sched_watchdog.go`, `server/concurrency_groups.go`, `server/images.go`.

### Prompt truncation in responses

When input exceeds effective `num_ctx`, final `/api/chat` and `/api/generate` responses (and the last stream chunk) include:

| Field | Meaning |
|-------|---------|
| `prompt_truncated` | Token-level shorten (Go `chatPrompt` tail-trim, runner trim, or **runtime context-shift detect**) |
| `original_prompt_tokens` | Size **before** truncation (prefer this over guessing from `prompt_eval_count`) |
| `messages_truncated` / `messages_dropped` | Oldest chat turns dropped in `chatPrompt` |

**Why:** Runners and llama-server logged truncation / context-shift while clients got a normal 200. Agents treated `prompt_eval_count ≈ num_ctx` as a soft hint instead of an explicit overflow.

**Jul 2026 fix:**

- Go now returns the **pre**-tail-truncate token count from `chatPrompt` (not only the runner's post-trim size).
- Runtime proxy path (`:8081` / `X-Zerollama-Runtime`) calls `detect_context_overflow` so sidecar responses also set `prompt_truncated` when the admit prompt exceeded `num_ctx`.

Soft companions (not sufficient alone): `prompt_eval_count` pinned near `num_ctx`; `done_reason: "length"` (also means `num_predict` exhausted).

Set `"truncate": false` for HTTP **400** instead of silent truncation on Go paths that honor it.

**Access log:** `inference response out` includes `prompt_tokens`, `original_tokens`, `truncated_tokens`, and `messages_dropped` when applicable — **why:** fleet logs should show prompt sizing without parsing JSON bodies.

**MLX tail truncate:** token-ID front-drop in `chatPrompt` before runner load; see [mlx-agent-prompts.md](./mlx-agent-prompts.md).

Code: `server/truncation.go`, `server/prompt.go`, `server/inference_access_log.go`, `runtime/llama_timings.py` (`detect_context_overflow`), `llm/server.go`, `runner/*/runner.go`.

---

## Training blocks inference (default on)

**What:** While training is active, holds weights on CUDA, or a job is running, Go can:

- Pause **new** ggml loads (`PauseNewLoads`)
- Evict loaded runners (`UnloadAllRunners`)
- Reject **runtime proxy** requests (`503` with clear message)

**Why inference-first by default:** interactive chat is the primary product surface; training on a consumer GPU is batch work that can wait or be deferred.

Env: `ZEROLLAMA_BLOCK_INFERENCE_DURING_TRAINING` (default on). See [gpu-training.md](./gpu-training.md).

---

## T6 — Training submit policy (Go)

### Idle-wait gate

**What:** When `ZEROLLAMA_TRAINING_WAIT_INFERENCE_IDLE=1`, new training submits are rejected (HTTP **409**) while:

- ggml scheduler has pending/active work, or (by default) **loaded** runners (`ZEROLLAMA_TRAINING_WAIT_GGML_LOADED`)
- Python runtime `/health` reports `waiting` / `running` / `llama_server: true`

**Why:** starting fine-tuning while a GGUF is loaded on the same GPU is the most common “training failed immediately” support case on single-GPU hosts.

**Why fail-closed on `/health` errors:** if we cannot see runtime load, assuming idle could start training into an occupied card (`ZEROLLAMA_TRAINING_WAIT_FAIL_CLOSED`, default on with idle-wait).

Code: `server/inference_workload.go`, guard in `x/trainingworker` via `SetInferenceSubmitGuard`.

### Priority on submit

| `priority` | Behavior | Why |
|------------|----------|-----|
| `normal` | Idle-wait applies | Default safe path |
| `low` / `batch` | Prefer defer queue when busy | Overnight / batch jobs should not spam 409 |
| `high` / `interactive` | Bypass idle-wait | Operator override when they know the GPU is theirs |

Request body / TCP: `priority`, `queue_on_busy`. HTTP: `POST /api/train/jobs`.

### Defer queue (`defer-*` job IDs)

**What:** When policy would return **409** and defer is enabled (`ZEROLLAMA_TRAINING_QUEUE_ON_BUSY=1`, or `queue_on_busy: true`, or `priority: low`), Go accepts the job into an in-memory queue instead:

| Reject reason | Defer requires |
|---------------|----------------|
| Inference backlog | `ZEROLLAMA_TRAINING_WAIT_INFERENCE_IDLE=1` |
| Outside allowed window | `ZEROLLAMA_TRAINING_ALLOWED_WINDOW` set (idle-wait not required) |
| Misconfigured window env | **No defer** — HTTP **503**, one-time warning log |

- Poll interval: `ZEROLLAMA_TRAINING_QUEUE_POLL_SECS` (default 5s)
- On idle: promotes to Python `training.py` and returns real `job_id` via `promotedJobId` on the defer record
- **Tombstone TTL:** `ZEROLLAMA_TRAINING_QUEUE_TOMBSTONE_SECS` (default 24h) — **why:** keep `defer-*` queryable for status after promotion without unbounded map growth
- **Retries:** `ZEROLLAMA_TRAINING_QUEUE_RETRY_MAX` / `RETRY_SECS` — **why:** transient Python errors during promotion should not drop batch work silently
- **Cancel:** `DELETE /api/train/jobs/defer-…` for **waiting** jobs only
- **List:** merged into `GET /api/train/jobs` (waiting only unless `ZEROLLAMA_TRAINING_QUEUE_LIST_ALL=1`)

**Why defer IDs stay queryable after promotion:** clients polling only `defer-*` still see `promotedJobId` and proxied Python status until tombstone eviction — save `promotedJobId` before TTL expires.

**Inference defer** extends the idle-wait gate; **window defer** is separate (see table above).

### Allowed window (night batch)

**What:** `ZEROLLAMA_TRAINING_ALLOWED_WINDOW=22:00-06:00` rejects training submit outside the window (HTTP **409**). `ZEROLLAMA_TRAINING_WINDOW_TZ` selects IANA timezone (default: local). Spans midnight when start > end.

| Override | Why |
|----------|-----|
| `priority: high` | Operator knows the GPU is theirs |
| Defer queue + `ZEROLLAMA_TRAINING_QUEUE_ON_BUSY=1` | Hold jobs until inside the window **and** inference idle |

Code: `server/training_window.go`, `envconfig/training_window.go`, `server/training_defer_queue.go`, `server/training_submit.go`.

**Invalid window:** typo in `ZEROLLAMA_TRAINING_ALLOWED_WINDOW` blocks all training submit (503) until fixed — avoids silently running without a night window.

### Tight host checklist

```bash
export OLLAMA_MAX_LOADED_MODELS=1
export ZEROLLAMA_TRAINING_WAIT_INFERENCE_IDLE=1
export ZEROLLAMA_TRAINING_QUEUE_ON_BUSY=1   # optional: queue instead of 409
export ZEROLLAMA_TRAINING_WAIT_GGML_LOADED=1 # default with idle-wait
# Multi-GPU: ZEROLLAMA_TRAINING_WAIT_GGML_LOADED=0 if ggml on GPU0 must not block train on GPU1
```

---

## Phase 11 / 13 — Python runtime admission and VRAM heuristics

**Phase 11 narrative (WHY, priorities, code map):** [phase11-runtime-admission.md](./phase11-runtime-admission.md).

### Admission and model swap

**What:** Python caps queue depth, coordinates GPU mutex with training handoff, swaps `llama-server` when `options.gguf` changes.

**Why in Python:** scheduling experiments belong next to the PA scheduler before APIs freeze in native code ([ROADMAP](./ROADMAP.md) layer ladder).

### Host RAM pre-check

**What:** Linux `/proc/meminfo` budget before mmap-heavy GGUF load.

**Why:** on tight hosts, host OOM during weight mmap fails with opaque errors; fail early with readable `LlamaServerError`.

### GPU VRAM pre-check (heuristic)

**What:** Before starting `llama-server`, estimate required VRAM vs free memory per GPU (TP bottleneck = min free across participating devices).

**Probe** (`ZEROLLAMA_RUNTIME_VRAM_PROBE`):

| Mode | Why |
|------|-----|
| `auto` | Try NVML (`nvidia-ml-py`), then `nvidia-smi` — fewer subprocess spawns when NVML works |
| `auto` + **shared embed** (`ZEROLLAMA_RUNTIME_SHARED_PYTHON=1`, training + runtime in one process) | Prefer **`nvidia-smi`** (subprocess wait releases the GIL). If `nvidia-smi` is missing, **skip** NVML probes (fail-open) and log once — see `docs/bugs/shared-interpreter-health-hang.md`. |
| `nvml` | Explicit NVML only |
| `nvidia-smi` | Explicit subprocess only |

**Unified-memory fallback:** when NVML returns NOT_SUPPORTED, use Linux `MemAvailable` (`probe: host-unified` in `/health`). **Why:** matches ggml behavior on iGPU-class devices where framebuffer memory is not reported.

**KV scale** (heuristic, not exact llama.cpp math):

1. Request `options.num_ctx`
2. Env `ZEROLLAMA_RUNTIME_VRAM_NUM_CTX`
3. GGUF `context_length` metadata
4. Baseline 4096 for scale ratio

**Layer scale (fallback):** `block_count` vs `ZEROLLAMA_RUNTIME_VRAM_LAYER_BASE` (32) when exact KV metadata is missing.

**Enqueue + dequeue VRAM pre-check:** when the GGUF path is known at submit time, the engine runs `check_gguf_host_budget` then `check_gguf_vram_budget` **before** the request enters the waiting queue (same skip rules as dequeue when the model is already loaded with sufficient `num_ctx`). Dequeue re-runs policy + budget when KV blocks are allocated—fail before llama-server start if free VRAM dropped while waiting. GPU budget uses **`resolve_vram_num_ctx`** (same as load: request `options.num_ctx` / env / `-c` / GGUF metadata), not only `req.num_ctx`. Partial tick admits roll back on failure. `/health` and `/internal/vram-estimate` expose `host_ram` (same `ZEROLLAMA_RUNTIME_RAM_MARGIN` as load) even when GPU free VRAM is unknown.

**Load `num_ctx` parity:** `resolve_num_ctx_for_request()` resolves options/env/`-c`, optionally clamps (`VRAM_CLAMP_NUM_CTX`), then `_admit_one` queues that ctx and `generate`/`stream_generate` load with **`active.num_ctx`** as `-c`. **Why:** uncapped client `num_ctx` must not reach `llama-server` after clamp/precheck.

**Opt-in context clamp:** `ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX` default **`0`**. `auto` or `1` (with GPU checks) lowers request `num_ctx` to `vram_budget.suggested_max_num_ctx` at enqueue. Responses include `vram_num_ctx` when clamped. **Why default off:** operators expect requested context unless they opt in.

**Operator pre-flight:** `scripts/runtime/runtime_vram_estimate.sh` → `POST /internal/vram-estimate` (loopback). Go proxy may call the same path asynchronously (`LogVramBudgetIfTight`). **Why:** choose quant/ctx before load; same math as `/health`.

**IQ/TQ KV sizing:** exact KV uses `max(ggml block bytes/element, F16)` for IQ/TQ/MXFP metadata types so estimates stay conservative when llama-server uses an F16 cache.

**Exact KV (preferred):** sum per layer: `ctx_eff(layer) × n_kv × (k_dim×k_bytes + v_dim×v_bytes)`; optional UINT32 `sliding_window` **array** (hybrid SWA); scalar `sliding_window` caps all layers. Unknown ggml KV types → fp16. `ZEROLLAMA_RUNTIME_VRAM_SCRATCH_FACTOR` (default 1.05) on weights+KV. Hints cached for `/health`. **Why:** add weights + KV instead of multiplying heuristics that already encode ctx and depth.

**llama-server flags in estimates:** `-c` / `--ctx-size` (after request `num_ctx` and `VRAM_NUM_CTX`), `-np` / parallel slots (YAML `llama_parallel_slots` default), `-ngl` scales weight bytes on GPU. Parsed from `RuntimeConfig.llama_server_args()` (includes `LLAMA_SERVER_EXTRA_ARGS`).

**Speculative draft:** for `draft-simple` / `draft-eagle3` / `draft-mtp`, add a second GGUF budget (`draft_model` in YAML or `--model-draft`; `--spec-draft-ngl` for draft layers). Ngram methods add no extra GGUF. Draft KV uses the same `num_ctx` (conservative).

**Admission:** **2 GiB** training reserve (default; `ZEROLLAMA_RUNTIME_TRAINING_VRAM_RESERVE`) subtracted at enqueue and load when handoff, non-`RUNNING`, or Go `training-gpu-busy`. **1 GiB** minimum free GPU when `CHECK_GPU_VRAM` is on (`ZEROLLAMA_RUNTIME_VRAM_MIN_FREE`); `priority: high` bypasses that gate; `low`/`batch` needs **1.5×** headroom. Heuristic VRAM path applies `-np` to KV only, not weights.

**Dequeuing:** `SchedulerLoop.tick` re-runs VRAM admission before KV block allocation so a full queue does not drain into OOM when free memory dropped during wait. `AdmissionRejected` re-queues; `AdmissionMisconfigured` propagates. Failed `_admit_one` / batch paths call `cancel_waiting` so orphans do not linger. Inference-first **dequeue** stalls only when the waiting **head** is `priority: low` (same mirrors as enqueue: backlog, defer, ggml, cross-FIFO); **`normal` chat keeps dequeuing** under those signals. **`high`** is always taken from the front when it is head.

**mmap:** optional `ZEROLLAMA_RUNTIME_VRAM_MMAP_FACTOR` (0–1) scales weight bytes in estimates when not all tensors are GPU-resident (default **1.0** = conservative).

**Partial GPU offload (`-ngl`):** with `VRAM_WEIGHT_TENSOR=1` (default), VRAM estimates sum GGUF tensors for layers `blk.0` … `blk.(ngl-1)` plus global weights. `VRAM_WEIGHT_BLOCK_LAYOUT=1` uses ggml block sizes for Q4_K, **IQ**, TQ, and MXFP4 types; `VRAM_KV_BLOCK_LAYOUT=1` (default) applies the same layout to **IQ/TQ/MXFP** KV cache types in exact KV estimates (legacy Q4/Q8 KV metadata still counts ≥2 bytes/element) (`gguf_estimate._GGML_BLOCK_LAYOUT`, synced from ggml-common.h). `ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR` multiplies the final estimate for operator calibration. **`VRAM_PROBE_CALIBRATE=auto`** records fresh free VRAM before/after llama-server load (bypasses 250ms probe cache); `/health` `vram_calibration.suggested_estimate_factor` ≈ observed/raw — **set** `VRAM_ESTIMATE_FACTOR` to that value (replace, not multiply; `estimated_effective_bytes` on the snapshot aligns to raw × suggested). **`VRAM_ESTIMATE_FACTOR_AUTOTUNE=auto`** applies the last suggested factor for pre-checks (per resolved GGUF path); **`VRAM_AUTOTUNE_PERSIST=auto`** writes `vram_autotune.json` v2 under `ZEROLLAMA_RUNTIME_STATE_DIR` (default `~/.cache/zerollama`) so restarts reuse the matching model’s factor. **`VRAM_ESTIMATE_FACTOR_EXPORT=auto`** writes `vram_estimate_factor.env` (sourceable `export ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR=…`) and `vram_autotune_factors.env` (per-model catalog) after each probed load. **`VRAM_APPLY_EXPORTED_ENV=1`** loads that file at runtime startup when `VRAM_ESTIMATE_FACTOR` is unset (`VRAM_APPLY_EXPORTED_ENV_PATH` to override); autotune persist still wins per GGUF. Requires probe calibration to refresh entries. v1 single-factor JSON files are migrated on read. `gguf_weight_bytes` uses `max(Σ tensor metadata, on-disk tensor region)`. Exact KV sums only GPU layers. `-ngl 0` → no GPU weights and no GPU KV.

**Coordination without training monitor:** when `ZEROLLAMA_RUNTIME_URL` is set but `BlockInferenceDuringTraining` is off (or training disabled), Go runs `runRuntimeCoordinationPusher` (~400ms) so defer/ggml mirrors stay fresh.

**Ggml pause when runtime busy (Go):** `ZEROLLAMA_GGML_PAUSE_WHEN_RUNTIME_BUSY=auto` (default on when a runtime is configured — external `ZEROLLAMA_RUNTIME_URL` or embedded sidecar) calls `sched.PauseNewLoads()` while `runtime_waiting + runtime_running` ≥ `ZEROLLAMA_GGML_PAUSE_RUNTIME_MIN_BACKLOG` (default 4). Mirror includes `ggml_loads_paused`. Complements Python inference-first (runtime rejects batch/low when ggml is busy).

**Go training GPU busy:** `server/training_policy.go` calls `POST /internal/training-gpu-busy` on the runtime when `trainingOccupiesGPU` toggles. Python applies the **2 GiB** training reserve (code constant) while busy. `/health` admission snapshot includes `go_training_gpu_busy`.

**Go coordination mirror:** `POST /internal/go-coordination` pushes `defer_waiting`, `defer_tracked`, `training_gpu_blocked`, `block_inference_during_training`, `ggml_loads_paused`, and **ggml workload** (`sched_pending`, `sched_active`, `sched_loaded`, optional `runtime_waiting` / `runtime_running` / `runtime_llama_loaded`) to runtime `/health` → `go_coordination`. One runtime `/health` GET per policy tick (training monitor or coordination pusher, ~400ms). Snapshots older than `ZEROLLAMA_RUNTIME_GO_COORDINATION_TTL_S` (default **30s**) are **stale** and ignored for cross-queue admission (fail-open). Go logs one warning if pushes fail.

**Defer / ggml / runtime backlog (inference-first):** when the Go mirror is **fresh**, batch (`priority: low`) is rejected at enqueue and stalled at dequeue if defer waiting ≥1, ggml sched backlog ≥1, ggml loads paused, or local runtime backlog ≥4. **`priority: normal`** is not blocked by those mirrors (VRAM checks still apply). **`priority: high`** bypasses min-free VRAM gate and is dequeued ahead of other work. Stale mirror → fail-open on mirrored metrics.

**Cross-queue FIFO (T6):** Go allocates monotonic tickets (`POST /internal/cross-queue-seq`, loopback-only). Python uses `ZEROLLAMA_GO_URL` or `connectable_go_base_url()` (maps `OLLAMA_HOST` `0.0.0.0` → `127.0.0.1`). `zerollama serve` sets `ZEROLLAMA_GO_URL` from `envconfig.ConnectableHost()` when unset. Mirror: `fifo_go_oldest_ggml` (pending + in-flight load), `fifo_go_oldest_defer`, `fifo_go_oldest_inference` (min of ggml + mirrored runtime), `fifo_runtime_oldest` on runtime `/health` (waiting **and running**). Python batch/low blocks when **ggml** ticket &lt; runtime head (defer uses separate gates). Ggml `pendingPopNext` yields while runtime is ahead; defer promotion waits while **inference** (runtime or ggml) is ahead. `useLoadedRunner` / model-prefer pop are not ticket-ordered.

**Shutdown:** training monitor and coordination pusher call `finalizeInferenceCoordination` on exit — `ResumeLoads` plus a final `go-coordination` push with `ggml_loads_paused=false` so mirrors do not linger until TTL.

**VRAM reserve vs mirror:** training reserve uses `POST /internal/training-gpu-busy` (authoritative). `go_coordination.training_gpu_blocked` is for `/health` only.

**Runtime VRAM admission:** follows `ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM` (same as load pre-check and enqueue/dequeue GGUF budget). **1 GiB** min free (constant, folded into model budget checks); **2 GiB** training reserve while training holds GPU. Probe unavailable → **503** (fail closed). `options.priority: high` bypasses the min-free gate (not model fit at load); `low`/`batch` needs **1.5×** min free. High priority is admitted at the **front** of the runtime waiting deque.

**Inference-first vs VRAM:** `ZEROLLAMA_RUNTIME_INFERENCE_POLICY=off` disables defer/ggml/runtime backlog throttling for batch (`low`) only; `CHECK_GPU_VRAM=1` still runs host + GGUF budget checks.

**Operator env (Python runtime):** `ZEROLLAMA_RUNTIME_INFERENCE_POLICY=off`, `ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM=0` — policy surface; backpressure thresholds live in code. **`ZEROLLAMA_RUNTIME_VRAM_MIN_FREE`**, **`ZEROLLAMA_RUNTIME_TRAINING_VRAM_RESERVE`** (size strings; prefer **GiB**). **`ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX`** (default off). Optional **`ZEROLLAMA_RUNTIME_MAX_QUEUE`** (default **512**) is a waiting-queue safety cap only. See [phase13-runtime-vram.md](./phase13-runtime-vram.md).

**`/health` `admission.gates_active`:** `true` means the signal is on. Only **`priority: low`** is rejected at enqueue or stalled at dequeue when it is queue head. **`runtime_backlog_pressure`** (backlog ≥ 4) does **not** block `normal` / `high`. Older dashboard keys are mirrored under `gates_active_compat`.

Env table: [runtime/docs/OPERATIONS.md](../runtime/docs/OPERATIONS.md).

**Why heuristics can reject good loads or allow bad ones:** margins (`ZEROLLAMA_RUNTIME_VRAM_MARGIN`, `VRAM_KV_FACTOR`) are operator-tunable; goal is actionable errors before subprocess start, not bit-exact accounting.

Code: `runtime/runtime/gpu_vram.py`, `runtime/runtime/gguf_estimate.py`, `runtime/runtime/vram_suggest.py`, `runtime/runtime/engine.py` (`resolve_num_ctx_for_request`).

---

## Phase 12 — Runtime default routing (Go)

**What:** Text-only local GGUF can proxy to Python when `ZEROLLAMA_RUNTIME` / URL is set; **tools** use runtime generic JSON prompt + output parser (models with `{`/`name`/`arguments` style). Thinking, vision, logprobs, embed, MLX stay on ggml.

**Why:** runtime path is faster to iterate for plain chat; per-model Go templates and builtin parsers still live on ggml until Q3/Q4 parity ships.

Code: `server/runtime_inference_routing.go`, `server/runtime_*_proxy.go`, `server/runtime_manifest.go` (`runtimeProxyOptions`, `runtimeV1ProxyOptions` — manifest `options.gguf` for Phase 13 on `/v1/chat/completions`). v1 requests with think/reasoning/logprobs/vision fall through to ggml (`v1ChatNeedsLegacyRunner`).

---

## CI regression (Phase 10)

**What:** `.github/workflows/zerollama-regression.yaml` runs Go `server` + `envconfig` + `trainingworker` tests and Python `runtime` pytest (no GPU required).

**Why:** two schedulers and embedded Python make it easy to break policy wiring without a cross-language gate.

**GPU operator smokes (labeled runner, not CI default):** [testing-smoke.md](./testing-smoke.md) — `gpu_smoke_all.sh`, optional `RUN_E2E_TOOLS=1`, `gpu_health_report.sh`, `runtime_vram_estimate.sh`. Phase 13 clamp/tools probes: [phase13-runtime-vram.md](./phase13-runtime-vram.md).

---

## What is still roadmap

| Item | Why not done |
|------|----------------|
| Single FIFO across inference + training | **Partial** — global tickets + cross-queue ordering; richer latency classes still roadmap |
| Training progress SSE ([T3](./ROADMAP.md)) | Bridge is JSON poll today |
| Family tool output parsers ([Phase 12](./ROADMAP.md)) | **Done** — runtime streams via Go `parse-tool-output/session` + `chunk` (same parsers as ggml) |
| Exact KV from tensor metadata ([Phase 13](./ROADMAP.md)) | **Partial** — `attn_k`/`attn_v` shapes infer head dims when metadata sparse; `/health` `vram_estimate` + `vram_budget`; clamp + `resolve_num_ctx_for_request` shipped — doc [phase13-runtime-vram.md](./phase13-runtime-vram.md) |
| Auth on `/api/train` ([T2](./ROADMAP.md)) | Same threat model as main API pending |
| Host capacity APIs (can-load, metrics, Retry-After) | **Phase A shipped** — [inference-wishlist-host.md](./inference-wishlist-host.md) |
| Host pin / propose / thrash dampen | **Phase B shipped** — broker-respecting pins; B0 ggml-empty; 503 on pin/runtime conflict; propose `serialize_required`; `stable_multi_model_swap` still false |
| True multi-GGUF stable swap | **Phase B+ deferred** — Python single-resident; Go soft-pin is interim |

**Readable config / residency:** `GET /api/status` → `inference.config` (+ `pins`); progressive-probe flags on `GET /api/version` → `zerollama.capabilities`. **Dry-run:** `POST /api/can-load` (runtime `confidence=exact`, ggml `heuristic`; fail closed if estimate missing; `already_loaded` is path-matched). **Propose / pin:** `POST /api/propose-load`, `POST /api/pin`. **Metrics:** `GET /api/metrics` (ggml + runtime proxy). Busy `503` includes `Retry-After` (queues, Metal, pin/runtime VRAM conflicts).

**Code map addenda:** `server/can_load.go`, `server/propose.go`, `server/pin.go`, `server/runtime_broker.go`, `server/metrics.go`, `server/empty_gen.go`, `server/inference_config.go`.

---

## Beyond one machine (fleet)

Everything above is **per zerollama node**. When agents talk to **many hosts**, add a **fleet management layer** that:

- Discovers nodes (directional: **mDNS** on LAN; static list in K8s)
- Routes to **warm models** (loaded + low queue) instead of random idle GPUs
- Relies on **stream progress** (`status`, `queue_depth`) for agent cancel-while-queued policy

The management node does **not** replace this document’s VRAM broker or local FIFOs—it **picks which node** should receive the request. See [fleet-scheduling.md](./fleet-scheduling.md) and [ROADMAP — Fleet scheduling](./ROADMAP.md#fleet-scheduling-multi-node).

---

## Code map

| Area | Path |
|------|------|
| VRAM broker | `server/vram/broker.go` |
| Training defer queue | `server/training_defer_queue.go`, `server/training_submit.go` |
| Inference workload gate | `server/inference_workload.go` |
| Training policy monitor | `server/training_policy.go` |
| Training HTTP | `server/training_api.go` |
| Embedded training | `x/trainingworker/`, `training.py` |
| Runtime VRAM | `runtime/runtime/gpu_vram.py` |
| Env helpers | `envconfig/config.go` |
