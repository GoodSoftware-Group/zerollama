# GPU training integration (Go + Python)

This document describes how **Ollama’s Go daemon** and **embedded CPython** run GPU training: public APIs, in-process Python, VRAM policy, and environment variables. It focuses on **why** the system is shaped this way, not only what each knob does.

---

## Goals (why this exists)

1. **One operator-facing process** — Users start Ollama once. Training does not require a second process on the public port. **Why:** fewer moving parts, consistent logging, predictable upgrades, and aligned lifecycle with inference.

2. **Go owns the wire** — HTTP (`/api/train/*`) and legacy TCP (`:9500` newline JSON) are implemented in Go so versioning, auth (when you add it), and policy live in one place. **Why:** Python is great for PyTorch; Go is better for long-lived network listeners and tight integration with the existing scheduler.

3. **Python owns the GPU for training** — Model load, LoRA/QLoRA, datasets, and the training loop stay in Python (repo-root **`training.py`**). **Why:** the ecosystem for fine-tuning (Transformers, PEFT, Accelerate, bitsandbytes) is Python-first; reimplementing in Go/Rust would delay value and duplicate maintenance.

4. **In-process Python (CGO)** — Go embeds CPython via `x/trainingworker/pyembed` (no `python3 -m …` subprocess, no gRPC, no UDS between processes). **Why:** one deployment unit, no `grpcio` sidecar, same OOM coordination with a direct C callback from Python into Go.

5. **VRAM: inference-first (v1)** — On a single consumer GPU, inference and training contend for the same memory. **Why:** default policy favors **interactive inference**: when training hits CUDA OOM, Go **pauses new loads**, **evicts loaded inference runners**, then calls back into Python to **ack** so `load_model` may retry. Training mid-loop does not magically continue; **load_model** can wait once after relief; failed jobs still fail—**why:** safe default without pretending we can checkpoint-resume every arbitrary training graph.

---

## Architecture

**Queues (today):** **Inference** (generate/chat/embed routes → Go scheduler and/or Python runtime scheduler) and **training** (`/api/train/*`, embedded job processor) are **not one FIFO**—they are coordinated through VRAM handoff and OOM bridges. Full policy rationale: [scheduling-vram-policy.md](./scheduling-vram-policy.md). Optional **`ZEROLLAMA_TRAINING_WAIT_INFERENCE_IDLE=1`** rejects training submit while inference backlog is non-zero (see env table). With **`ZEROLLAMA_TRAINING_QUEUE_ON_BUSY=1`** (or `queue_on_busy` / `priority: low` on submit), policy rejects become **`defer-*` jobs** (inference backlog when idle-wait is on; outside **`ZEROLLAMA_TRAINING_ALLOWED_WINDOW`** when that env is set). On a single GPU, a loaded runtime GGUF (`llama_server: true` in `/health`) and resident ggml models (`ZEROLLAMA_TRAINING_WAIT_GGML_LOADED`, default on with idle-wait) count as busy — with a long **`OLLAMA_KEEP_ALIVE`**, training may be denied until models expire or you unload them. Set **`ZEROLLAMA_TRAINING_WAIT_GGML_LOADED=0`** if keep-alive ggml on one GPU should not block training on another. Finer priority classes and cross-queue SLOs remain roadmap [T6](./ROADMAP.md#phases-training-track).

```
Clients                    Ollama (Go)                              CPython (embedded)
───────                    ─────────────                              ──────────────────
HTTP /api/train/*  ─────►  Gin handlers ──CGO──►  pyembed (libpython)   training.py + JOB_QUEUE
TCP :9500 JSON     ─────►  trainingworker.ServePublicTCP              job_processor thread
```

- **Public TCP 9500** is bound **only by Go** (`x/trainingworker`). Legacy newline-JSON clients are unchanged.

- **Python** does not listen on the network for training control; it runs inside the Go process. **Why:** single security boundary for public surfaces.

---

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_TRAINING` | `true` | When `false`, no embedded interpreter, no `/api/train` routes, no `:9500` listener. **Why default on:** feature is discoverable for integrators; production can opt out explicitly. |
| `OLLAMA_TRAINING_TCP` | `:9500` | Address for Go’s public training TCP listener. `0` or `-` disables TCP (HTTP-only). **Why:** some deployments want training HTTP only or a different bind address. |
| `OLLAMA_TRAINING_PYTHONPATH` | (auto) | Repository root directory that contains **`training.py`**. **Why:** installed binaries may not sit next to the repo; this makes layout explicit. When unset, resolution also walks **cwd** upward and checks **`$HOME/zerollama`** and **`$HOME/ollama`**. Alias: **`ZEROLLAMA_REPO`**. |
| `TRAINING_WORKER_IDLE_UNLOAD_SEC` | `300` | In `training.py`: seconds after a job before unloading the cached model from GPU (`0` = off). **Why:** frees VRAM between sparse training sessions without killing throughput for back-to-back jobs. |
| `ZEROLLAMA_BLOCK_INFERENCE_DURING_TRAINING` | on | While training holds the GPU (active job, queue running, or HF model loaded), block runtime proxy inference and evict ggml + `llama-server`. Set `0` to disable. |
| `ZEROLLAMA_TRAINING_WAIT_INFERENCE_IDLE` | off | When `1`, reject new training jobs (HTTP `/api/train/jobs` and TCP `submit_job`) until ggml scheduler and runtime health (`waiting`/`running`/`llama_server`) are idle. **Why:** inference-first batch scheduling ([T6](./ROADMAP.md#phases-training-track)). |
| `ZEROLLAMA_TRAINING_WAIT_FAIL_CLOSED` | on when idle-wait on | When idle-wait is enabled, treat unreachable runtime `GET /health` as busy (reject submit). Set `0` to allow submit if the probe fails but ggml is idle. |
| `ZEROLLAMA_TRAINING_WAIT_GGML_LOADED` | on when idle-wait on | Count resident ggml runners (`OLLAMA_KEEP_ALIVE`) as busy. Set `0` on multi-GPU hosts if training may use another card while models stay loaded. |
| `ZEROLLAMA_TRAINING_QUEUE_ON_BUSY` | off | Queue instead of HTTP **409** when policy rejects: **(1)** inference backlog — requires **`ZEROLLAMA_TRAINING_WAIT_INFERENCE_IDLE=1`**; **(2)** outside **`ZEROLLAMA_TRAINING_ALLOWED_WINDOW`** — window env alone is enough. Invalid window string logs a warning and returns **503** (fail closed, no defer). Poll interval: `ZEROLLAMA_TRAINING_QUEUE_POLL_SECS` (default 5). Max depth: `ZEROLLAMA_TRAINING_QUEUE_MAX` (default 32). Tombstone TTL: `ZEROLLAMA_TRAINING_QUEUE_TOMBSTONE_SECS` (default 86400; `0` = keep forever). Retries: `ZEROLLAMA_TRAINING_QUEUE_RETRY_MAX` (default 3), `ZEROLLAMA_TRAINING_QUEUE_RETRY_SECS` (default 30). List merge: `ZEROLLAMA_TRAINING_QUEUE_LIST_ALL=1` includes terminal defer jobs in `GET /api/train/jobs` (default: **waiting only**). After promotion, keep polling the same `defer-*` id — status includes `promotedJobId` and proxies the Python job when available. |
| Request `priority` | `normal` | `high` / `interactive` bypass idle-wait and allowed window. `low` / `batch` prefers defer queue when busy. Per-request `queue_on_busy: true` also enables defer without the env. |
| `ZEROLLAMA_TRAINING_ALLOWED_WINDOW` | off | e.g. `22:00-06:00` (must be `HH:MM-HH:MM`). Outside window → **409**; malformed value → **503** + one-time log (fail closed). `ZEROLLAMA_TRAINING_WINDOW_TZ` (IANA or `local`). With `ZEROLLAMA_TRAINING_QUEUE_ON_BUSY=1`, queue until inside window. |

---

## HTTP API (Go)

Base path: **`/api/train`** (only registered if the training client started successfully).

| Method | Path | Notes |
|--------|------|------|
| `POST` | `/api/train/jobs` | Async job; body `{"kind":"train"|"run_script","payload":{...},"priority":"normal"|"low"|"high","queue_on_busy":true}` |
| `GET` | `/api/train/jobs` | List recent jobs (JSON, same field names as historical protobuf JSON) |
| `GET` | `/api/train/jobs/:id` | Job status |
| `DELETE` | `/api/train/jobs/:id` | Cancel Python jobs or **waiting** `defer-*` jobs |
| `POST` | `/api/train/unload` | Unload training model on Python side |
| `GET` | `/api/train/status` | Health + queue extras |

**Why** a separate HTTP surface: modern clients prefer REST over raw TCP; the TCP path remains for backward compatibility.

---

## Legacy TCP protocol (Go)

Same newline-delimited JSON as the historical worker: `ping`, `submit_job`, `job_status`, `list_jobs`, `cancel_job`, `queue_status`, `train`, `run_script`, `unload`, `shutdown`.

**Timeouts:** While waiting for the next line, Go uses a **read idle deadline**; deadlines are cleared before handling a request so **multi-hour synchronous `train`** is not cut off mid-job—**why:** a single connection-wide deadline would break long jobs.

---

## Internal bridge (Go ↔ embedded Python)

Not a public API. Go calls C (`training_shim.c`), which uses the Python C API and a small bootstrap script (`x/trainingworker/pyembed/bootstrap.py`) to:

- Install a `BridgeState` subclass (OOM hooks, same behavior as the former `gpu_session.py`).
- Expose JSON-shaped responses compatible with the old HTTP/proto JSON clients.

**Progress (polling, not push):** embedded mode does not stream fine-grained events to Go over a second channel. Clients observe training via **`GET /api/train/jobs/:id`** (or legacy TCP `job_status`). **Why:** the old gRPC daemon could push progress; the in-process bridge optimizes for a small surface (JSON in/out) and keeps long-running TCP `train` commands workable without a parallel event bus.

**OOM path:** `training.py` → `ollama_training_native.fire_oom` (C extension registered before `Py_Initialize`) → Go callback → scheduler eviction → `training_ack_vram_headroom` → Python `threading.Event`.

**Why no `Py_Finalize`:** finalizing the interpreter after `torch` has been imported is unsafe; shutdown only stops the job thread and clears state; process exit reclaims resources.

---

## VRAM coordination (OOM bridge)

1. Python detects CUDA OOM, calls **`fire_oom`** on the native module (still holding GIL briefly).
2. C releases the GIL and invokes the Go callback: `PauseNewLoads` → `UnloadAllRunners` → **`AckVRAMHeadroom`** (into Python) → **`defer ResumeLoads`**.
3. **Why `defer ResumeLoads`:** if eviction panics or returns early, inference must not stay paused forever.
4. **Why `PauseNewLoads`:** avoids a new chat load grabbing VRAM between eviction and training retry.

Python **`load_model`** can wait once and retry after ACK. Mid-training OOM still fails the job but **notifies** Go so VRAM is saner for the **next** job—**why:** restarting an arbitrary Trainer mid-epoch without checkpoints is unsafe; we do not fake success.

---

## OOM synchronization in Python (ordering and pitfalls)

For paths that **wait** for Go (e.g. `load_model` retry, `job_processor` outer OOM), Python must follow a strict order:

1. **`_prepare_vram_relief_wait`** — register a `threading.Event` for the current `job_id` *before* any code path can call into Go. **Why:** Go may ack VRAM relief very quickly; if the event did not exist yet, the ack would be a no-op and Python would block until timeout.
2. **`_notify_cuda_oom`** — `fire_oom` into Go (eviction + ack).
3. **`_wait_vram_relief_after_oom`** — wait on the **same** event object prepared in step 1. **Why:** creating a *second* event in `_wait` (after Go already ack’d the first) caused a **lost wakeup** and up to a 120 second stall.

**Mid-training OOM** (`process_training_request` inner `except`): only step 2 runs. **Why:** that job is not retried in place; there is nothing to unblock. Go’s ack still runs (`ack_vram_headroom` pops nothing) so inference VRAM is freed for the **next** job without risking a bogus wait on a never-registered event.

**Shutdown:** `shutdown_ollama_training` sets `STATE.running = false` and signals `_pending_oom_event` if set so a thread blocked in step 3 can exit before `join` times out. **Why:** the job thread is otherwise allowed up to 120s on the OOM wait.

---

## Lifecycle

- **Start:** Go runs `Py_Initialize`, loads bootstrap + `training.py`, starts `job_processor`, then `PyEval_SaveThread` so inference and Python threads can run concurrently. **`PyThreadState*` from `SaveThread` is intentionally discarded** — later call-ins use `PyGILState_*`; pairing `PyEval_RestoreThread` at shutdown is unsafe with torch + Go.
- **Stop:** On shutdown, Go calls `shutdown_ollama_training()` in Python (join job thread). **No** `Py_Finalize` after torch import. If `training_init` fails after `Py_Initialize`, the process must **restart** to retry (`g_init_aborted`); **why:** we do not call `Py_Finalize` to unwind a half-imported stack.

---

## Python layout

- Repo-root **`training.py`** — job queue, `WorkerState`, training loop.
- **`x/trainingworker/pyembed/bootstrap.py`** — loaded as a string at init; wires `BridgeState` and the native OOM module. Not a standalone installable package.

**Dependencies:** same as before for `training.py` (`torch`, `transformers`, `peft`, …). **`grpcio` is not required** for training control.

### Apple Silicon (MPS)

On **darwin**, the same embedded **`training.py`** worker runs with **PyTorch MPS** when CUDA is unavailable. Outputs match the CUDA path: **`lora_adapter/`** with PEFT **`adapter_model.safetensors`**, **`adapter_config.json`**, and tokenizer files — import via Modelfile **`ADAPTER`**.

| Feature | CUDA | Apple Silicon (MPS) |
|---------|------|------------------------|
| LoRA (`use_lora: true`) | Yes | Yes |
| QLoRA (`use_qlora: true`) | Yes (bitsandbytes) | **No** — rejected with a clear error |
| VRAM OOM bridge (Go eviction) | Yes | Best-effort; unified memory, no NVML |
| Training dtype | `bf16` | `float32` |

Install deps with **uv** into **`.venv-training`** at the repo root (same pattern as `runtime/.venv`):

```bash
./scripts/training_uv_venv.sh --verify
# several uv binaries on PATH:
UV_BIN="$HOME/.local/bin/uv" ./scripts/training_uv_venv.sh --verify
eval "$(./scripts/training_uv_venv.sh --export)"
zerollama serve
```

Or use `./scripts/serve_mac_runtime.sh` — it picks up `.venv-training` automatically when present.

Submit jobs the same way as on Linux (`POST /api/train/jobs` with `kind: train`). Use **`use_lora: true`** and **`use_qlora: false`**.

### Installing Python deps (embedded interpreter)

`zerollama` embeds **system** `libpython3` (check with `otool -L` on Mac or `ldd` on Linux). Training packages are loaded via **`PYTHONPATH`** pointing at the uv venv site-packages — not by replacing the embedded interpreter.

**Recommended (uv):**

```bash
./scripts/training_uv_venv.sh --verify
eval "$(./scripts/training_uv_venv.sh --export)"
```

**Env:**

| Variable | Purpose |
|----------|---------|
| `UV_BIN` | Which `uv` to use when several copies are installed |
| `TRAINING_UV_VENV` | Override venv path (default: `$REPO/.venv-training`) |
| `TRAINING_UV_AUTO=1` | `serve_mac_runtime.sh` creates the venv on first start if missing |

**Linux + NVIDIA** (CUDA index passed by the script on non-Darwin):

```bash
UV_BIN=uv ./scripts/training_uv_venv.sh --verify
eval "$(./scripts/training_uv_venv.sh --export)"
```

After a failed `training_init`, **restart** the daemon (`g_init_aborted` blocks retry in-process).

When `OLLAMA_HOST` binds `0.0.0.0`, set `ZEROLLAMA_GO_URL=http://127.0.0.1:8080` for embedded runtime → Go `/internal/*` (or rely on serve auto-setting it).

**Smoke:** `OLLAMA_HOST=http://127.0.0.1:8080 ./scripts/e2e_training_ops_smoke.sh` and  
`RUN_E2E_TRAINING_TCP=1` (TCP uses `{"cmd":"ping"}` on `OLLAMA_TRAINING_TCP`, default `:9500`).

---

## Build / link requirements

- **CGO enabled** (`CGO_ENABLED=1`, default on many platforms).
- **`pkg-config python3-embed`** (from `python3-dev` / `python3-devel`) so the linker finds `libpython3`.

---

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| `Python.h: No such file` at build | Install `python3-dev` / `python3-devel` and `pkg-config`. |
| `does not contain training.py` | `OLLAMA_TRAINING_PYTHONPATH` or `ZEROLLAMA_REPO` is set but wrong—fix the path (no auto-fallback). |
| `set OLLAMA_TRAINING_PYTHONPATH` | No `training.py` found via discovery (binary walk, cwd walk, `$HOME/zerollama`). Example: [`scripts/serve_gpu_example.sh`](../scripts/serve_gpu_example.sh). |
| `training_init failed` / import errors | Missing Python deps (`torch`, …). Run [`./scripts/training_uv_venv.sh --verify`](../scripts/training_uv_venv.sh). |
| `No module named 'torch'` with `(.venv)` active | Training uses `PYTHONPATH` from `.venv-training`, not an activated shell venv. Run `./scripts/training_uv_venv.sh --export` before `serve`. |
| `Failed to load PyTorch C extensions` / `torch/_C` folder | **Python ABI mismatch:** Go embeds system `libpython3-embed` (often 3.10) but `.venv-training` was built for 3.11. Recreate: `TRAINING_UV_PYTHON_VER="$(pkg-config --modversion python3-embed)" ./scripts/training_uv_venv.sh --verify` then rebuild/restart. Shim auto-picks `.venv-training/lib/pythonX.Y/site-packages` matching embedded version. |
| `embedded Python failed to start earlier; restart the process` | A prior `training_init` failed after `Py_Initialize`; `g_init_aborted` is set. **Why:** we do not `Py_Finalize` a half-started interpreter (unsafe with torch). |
| Port 9500 in use | Set `OLLAMA_TRAINING_TCP` to another address or disable with `0`. |
| Embedded runtime `GET /health` hangs (training on) | Known shared-interpreter issue: [`docs/bugs/shared-interpreter-health-hang.md`](bugs/shared-interpreter-health-hang.md), repro [`scripts/repro_shared_interpreter_health_hang.sh`](../scripts/repro_shared_interpreter_health_hang.sh). Workaround: `OLLAMA_TRAINING=false` or external runtime (`ZEROLLAMA_RUNTIME_URL` + `ZEROLLAMA_RUNTIME_EMBED=0`). |
| `free(): invalid pointer` at training step 0 (5080) | Mismatched libcudnn: serve sets `/usr/hostlibs` for ggml while embedded torch needs bundled cuDNN in `.venv-training/.../torch/lib`. **Fix:** rebuild `zerollama` so `training_shim.c` runs `embedded_prepare_pytorch_ld_path()` before `Py_Initialize` ([`x/pyembed_common/embedded_pytorch_env.c`](../x/pyembed_common/embedded_pytorch_env.c)). Confirm log line `prepended .../torch/lib to LD_LIBRARY_PATH`. Ensure `eval "$(./scripts/training_uv_venv.sh --export)"` before serve. |

---

## Related files (code map)

| Area | Location |
|------|----------|
| Go training client + TCP bridge | `x/trainingworker/client.go` — see [`x/trainingworker/README.md`](../x/trainingworker/README.md) |
| CGO + C shim + bootstrap | `x/trainingworker/pyembed/` — see [`pyembed/README.md`](../x/trainingworker/pyembed/README.md) |
| Shared-interpreter `/health` hang | [`docs/bugs/shared-interpreter-health-hang.md`](bugs/shared-interpreter-health-hang.md) |
| HTTP handlers | `server/training_api.go` |
| Serve wiring | `server/routes.go` |
| Scheduler hooks | `server/sched.go` (`PauseNewLoads`, `UnloadAllRunners`, `ResumeLoads`) |
| Training logic | `training.py` |

---

## Further reading

- [Roadmap — GPU training](ROADMAP.md#gpu-training-fine-tuning) — planned improvements and non-goals.
- [CHANGELOG](../CHANGELOG.md) — when this landed and notable fixes.
