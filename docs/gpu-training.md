# GPU training integration (Go + Python)

This document describes how **Ollama’s Go daemon** and a **Python GPU training daemon** work together: public APIs, internal RPC, VRAM policy, and environment variables. It focuses on **why** the system is shaped this way, not only what each knob does.

---

## Goals (why this exists)

1. **One operator-facing process** — Users start Ollama once. Training must not require a second manually managed server on the public port. **Why:** fewer moving parts, consistent logging, predictable upgrades, and aligned lifecycle with inference.

2. **Go owns the wire** — HTTP (`/api/train/*`) and legacy TCP (`:9500` newline JSON) are implemented in Go so versioning, auth (when you add it), and policy live in one place. **Why:** Python is great for PyTorch; Go is better for long-lived network listeners and tight integration with the existing scheduler.

3. **Python owns the GPU for training** — Model load, LoRA/QLoRA, datasets, and the training loop stay in Python (reusing `training.py` as a library). **Why:** the ecosystem for fine-tuning (Transformers, PEFT, Accelerate, bitsandbytes) is Python-first; reimplementing in Go/Rust would delay value and duplicate maintenance.

4. **Strong internal IPC** — Go talks to Python over **gRPC on a Unix domain socket (UDS)**, not ad-hoc JSON on a shared TCP port. **Why:** typed contracts (`proto/training.proto`), streaming (progress, OOM), easier evolution than newline-delimited JSON between processes.

5. **VRAM: inference-first (v1)** — On a single consumer GPU, inference and training contend for the same memory. **Why:** default policy favors **interactive inference**: when training hits CUDA OOM, Go **pauses new loads**, **evicts loaded inference runners**, then tells Python it may retry. Training mid-loop does not magically continue; **load_model** can retry once after relief; failed jobs still fail—**why:** safe default without pretending we can checkpoint-resume every arbitrary training graph.

---

## Architecture

```
Clients                    Ollama (Go)                         Python (trainingdaemon)
───────                    ─────────────                         ─────────────────────
HTTP /api/train/*  ─────►  Gin handlers ──gRPC──►  UDS : private   training.py (library)
TCP :9500 JSON     ─────►  trainingworker.ServePublicTCP         JOB_QUEUE + job thread
```

- **Public TCP 9500** is bound **only by Go** (`x/trainingworker`). Legacy clients that used to speak to `training.py` directly keep the same newline-JSON command set; Go translates to gRPC.

- **Python** listens **only** on the UDS path passed as `--uds-path`. It must not bind `:9500`—**why:** avoids port conflicts and keeps a single security boundary for “who speaks to the outside world.”

---

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_TRAINING` | `true` | When `false`, no Python subprocess, no `/api/train` routes, no `:9500` listener. **Why default on:** feature is discoverable for integrators; production can opt out explicitly. |
| `OLLAMA_TRAINING_TCP` | `:9500` | Address for Go’s public training TCP listener. `0` or `-` disables TCP (HTTP-only). **Why:** some deployments want training HTTP only or a different bind address. |
| `OLLAMA_TRAINING_PYTHONPATH` | (auto) | Directory that contains the `trainingdaemon` package (typically `…/x/trainingdaemon`). **Why:** installed binaries may not sit next to the repo; this makes layout explicit. |
| `TRAINING_WORKER_IDLE_UNLOAD_SEC` | `300` | In `training.py`: seconds after a job before unloading the cached model from GPU (`0` = off). **Why:** frees VRAM between sparse training sessions without killing throughput for back-to-back jobs. |

---

## HTTP API (Go)

Base path: **`/api/train`** (only registered if the training client started successfully).

| Method | Path | Notes |
|--------|------|------|
| `POST` | `/api/train/jobs` | Async job; body `{"kind":"train"|"run_script","payload":{...}}` |
| `GET` | `/api/train/jobs` | List recent jobs (protobuf JSON) |
| `GET` | `/api/train/jobs/:id` | Job status |
| `DELETE` | `/api/train/jobs/:id` | Cancel |
| `POST` | `/api/train/unload` | Unload training model on Python side |
| `GET` | `/api/train/status` | Health + queue extras |

**Why** a separate HTTP surface: modern clients prefer REST over raw TCP; the TCP path remains for backward compatibility.

---

## Legacy TCP protocol (Go)

Same newline-delimited JSON as the historical worker: `ping`, `submit_job`, `job_status`, `list_jobs`, `cancel_job`, `queue_status`, `train`, `run_script`, `unload`, `shutdown`.

**Why** Go implements this: migration without rewriting every client; internal representation can still be protobuf.

**Timeouts:** While waiting for the next line, Go uses a **read idle deadline**; deadlines are cleared before handling a request so **multi-hour synchronous `train`** is not cut off mid-job—**why:** a single connection-wide deadline would break long jobs.

---

## Internal gRPC (`proto/training.proto`)

Not stable as a public API; it is the **Go ↔ Python** contract. Key RPCs:

- `Health`, `SubmitJob`, `JobStatus`, `ListJobs`, `CancelJob`, `Unload`, `Shutdown`
- `StreamEvents` — progress, completion, failure, **OOM** (for the bridge below)
- `AckVRAMHeadroom` — Go tells Python “we evicted inference; you may retry allocation”

**Why `AckVRAMHeadroom`:** OOM notification alone is racy; Python registers a wait **before** emitting OOM so an early ACK cannot be lost.

---

## VRAM coordination (OOM bridge)

1. Python detects CUDA OOM (or message-shaped OOM), emits an **OOM event** on the stream (with `job_id`).
2. Go’s `runOOMBridge`: `PauseNewLoads` → `UnloadAllRunners` → `AckVRAMHeadroom` → **`defer ResumeLoads`**.
3. **Why `defer ResumeLoads`:** if eviction panics or returns early, inference must not stay paused forever.
4. **Why `PauseNewLoads`:** avoids a new chat load grabbing VRAM between eviction and training retry.

Python **`load_model`** can wait once and retry after ACK. Mid-training OOM still fails the job but **notifies** Go so VRAM is saner for the **next** job—**why:** restarting an arbitrary Trainer mid-epoch without checkpoints is unsafe; we do not fake success.

---

## Lifecycle

- **Start:** Go runs `python3 -m trainingdaemon --uds-path <tmp.sock>`, sets `PYTHONPATH`, dials UDS until `Health` succeeds.
- **Stop:** On shutdown, Go calls `Shutdown` RPC, closes gRPC, waits for the process (with kill timeout), removes the socket. Python’s servicer stops the gRPC server so `wait_for_termination` can return—**why:** clean exit instead of orphan processes.

---

## Python package layout

- `x/trainingdaemon/` — installable-ish layout: `trainingdaemon` package, `pyproject.toml`, generated `training_pb2*.py`.
- Repo-root **`training.py`** — imported by the daemon; contains queue, `WorkerState`, and training logic.

See **`x/trainingdaemon/README.md`** for venv and dependencies.

---

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| `training worker: python3 not found` | Install Python 3 on `PATH`. |
| `set OLLAMA_TRAINING_PYTHONPATH` | Daemon not found next to binary; set env to `…/x/trainingdaemon`. |
| `training worker: dial` / daemon not ready | Missing deps (`torch`, `grpcio`, …), import error in `training.py`, or socket permission. Check stderr from the child process. |
| Port 9500 in use | Set `OLLAMA_TRAINING_TCP` to another address or disable with `0`. |

---

## Related files (code map)

| Area | Location |
|------|----------|
| Proto | `proto/training.proto` |
| Go gRPC client + TCP bridge | `x/trainingworker/client.go` — see also [`x/trainingworker/README.md`](../x/trainingworker/README.md) |
| Go stubs | `x/trainingworker/trainingpb/` |
| HTTP handlers | `server/training_api.go` |
| Serve wiring | `server/routes.go` |
| Scheduler hooks | `server/sched.go` (`PauseNewLoads`, `UnloadAllRunners`, `ResumeLoads`) |
| Python gRPC server | `x/trainingdaemon/trainingdaemon/server.py` |
| Bridge to `training.py` | `x/trainingdaemon/trainingdaemon/gpu_session.py` |
| Training logic | `training.py` |

---

## Further reading

- [Roadmap — GPU training](ROADMAP.md#gpu-training-fine-tuning) — planned improvements and non-goals.
- [CHANGELOG](../CHANGELOG.md) — when this landed and notable fixes.
