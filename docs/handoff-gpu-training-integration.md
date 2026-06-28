# Handoff: GPU training integration (Go + Python)

**Audience:** Another engineer with access to this repo who did not participate in the design and implementation thread.

**Purpose:** Capture **intent**, **decisions**, and **where to look next** for embedded GPU training and how it interacts with **Phase 11 runtime admission** on shared VRAM.

**Status (May 2026):**

| Item | State | Evidence |
|------|--------|----------|
| **Embedded CPython** | **Shipped** | `x/trainingworker/pyembed`; `trainingdaemon` + gRPC removed |
| **HTTP `/api/train/*`** | **Shipped** | `server/training_api.go` |
| **TCP `:9500`** | **Shipped** | `OLLAMA_TRAINING_TCP` in `x/trainingworker/client.go` |
| **OOM bridge** | **Shipped** | Go pause loads → unload runners → `AckVRAMHeadroom` |
| **Shared-interpreter `/health`** | **Mitigated** | `ZEROLLAMA_RUNTIME_SHARED_PYTHON=1`; repro script PASS (5× `/health`) |
| **CI** | **Green** | `go test ./x/trainingworker/...`; `e2e_training_ops_smoke.sh` in `check_gpu_scripts.sh` |

---

## What we were solving

We needed **GPU fine-tuning / training** integrated into the Ollama daemon without:

- A second public server that clients had to discover separately, or  
- Python binding the same TCP port as external training clients.

**Decision:** **Go** is the only process that listens on the **public** training surfaces (HTTP and, by default, **TCP `:9500`**). **CPython is embedded** in the Go process (CGO, `x/trainingworker/pyembed`), loads repo-root **`training.py`**, and uses the Python C API + a small bootstrap script for OOM/ack coordination—**no** separate `python3` training subprocess, **no** gRPC, **no** UDS control plane.

**Why embed instead of a sidecar:** Fewer moving parts at runtime (`grpcio` not required for control plane), one binary to ship (plus `libpython3` on the host), same VRAM story with a direct C callback from Python into Go.

**Why inference-first VRAM on OOM (v1):** Single-GPU setups contend for VRAM. When training signals CUDA OOM, Go **pauses new inference loads**, **unloads inference runners**, then **acks** Python so `load_model` can retry once. Mid-training-loop OOM still fails the job (no fake checkpoint resume).

---

## Interaction with Phase 11 (runtime admission)

Training and the Python runtime share one card. Phase 11 does **not** merge training and chat into one FIFO; it coordinates VRAM and queue pressure:

| Mechanism | Owner | Why |
|-----------|--------|-----|
| VRAM broker (Phase 8) | Go `server/vram` | Unload ggml + `training-handoff` on runtime before training submit |
| `POST /internal/training-gpu-busy` | Go `training_policy.go` → runtime | Python holds **2 GiB** reserve (`TRAINING_VRAM_RESERVE_BYTES`) while training occupies GPU |
| T6 defer queue + idle-wait | Go | Training submit rejected or deferred while inference busy |
| Inference-first (LOW only) | Python | Batch runtime work waits when defer/ggml mirrors busy; **normal chat not blocked** |

**Why `training-gpu-busy` is separate from `go-coordination`:** reserve must be authoritative when training holds CUDA; mirror `training_gpu_blocked` is for `/health` visibility only.

**Operator env (training):** see [gpu-training.md](./gpu-training.md).  
**Operator env (runtime admission):** only `INFERENCE_POLICY` + `CHECK_GPU_VRAM` — [phase11-runtime-admission.md](./phase11-runtime-admission.md).

When **embedded runtime + training** share one CPython (`ZEROLLAMA_RUNTIME_SHARED_PYTHON=1`), `/health` uses `nvidia-smi` for VRAM probe (GIL-friendly). See [bugs/shared-interpreter-health-hang.md](./bugs/shared-interpreter-health-hang.md).

---

## Conversation arc (compressed)

1. “Training on :9500” → Python for training, Go for public I/O.  
2. Embedded CPython instead of gRPC daemon.  
3. `OLLAMA_TRAINING` default on; `OLLAMA_TRAINING=false` when no torch stack.  
4. OOM bridge hardening (lost wakeup, shutdown, `ResumeLoads`).  
5. Phase 11 parallel track: opinionated runtime admission (no `ADMISSION_*` env sprawl).

---

## What to read first

| Doc | Role |
|-----|------|
| [gpu-training.md](./gpu-training.md) | Architecture, env vars, APIs, OOM bridge, troubleshooting |
| [phase11-runtime-admission.md](./phase11-runtime-admission.md) | Runtime VRAM + inference-first (shared GPU) |
| [scheduling-vram-policy.md](./scheduling-vram-policy.md) | Broker + T6 defer + full env tables |
| [pyembed/README.md](../x/trainingworker/pyembed/README.md) | CGO embed WHY |
| [handoff-phase12-runtime-tools.md](./handoff-phase12-runtime-tools.md) | Phase 12 tools + expanded Phase 11 handoff |
| [handoff-phase14-inprocess-llama.md](./handoff-phase14-inprocess-llama.md) | In-process forward + same Python process as training |
| [CHANGELOG.md](../CHANGELOG.md) | Unreleased |

---

## Code map

| Layer | Path |
|--------|------|
| Go: embed, TCP :9500 | `x/trainingworker/client.go` |
| Go: CGO shim + bootstrap | `x/trainingworker/pyembed/` |
| Go: HTTP `/api/train/*` | `server/training_api.go` |
| Go: training GPU policy | `server/training_policy.go` → `/internal/training-gpu-busy` |
| Go: VRAM broker | `server/vram/broker.go` |
| Go: wiring | `server/routes.go` |
| Go: scheduler hooks | `server/sched.go` |
| Env | `envconfig/config.go` |
| Training logic | `training.py` (repo root) |
| Runtime reserve consumer | `runtime/runtime/gpu/admission.py`, `engine.py` |

---

## How to sanity-check

1. **`python3-dev`** + **`pkg-config python3-embed`** for Go build. Training deps: `requirements-training.txt` / [gpu-training.md](./gpu-training.md).  
2. Start server with `OLLAMA_TRAINING=true` (default). Set `OLLAMA_TRAINING_PYTHONPATH` or `~/zerollama` layout. See [`scripts/serve_production_wrapper.sh`](../scripts/serve_production_wrapper.sh) → `~/bin/serve.sh`.  
3. `OLLAMA_HOST=http://127.0.0.1:8080 ./scripts/e2e_training_ops_smoke.sh` — `GET /api/train/status` + jobs list; optional `RUN_E2E_TRAINING_TCP=1`.  
4. **Shared embed:** `./scripts/repro_shared_interpreter_health_hang.sh` — 5× `/health` must not hang (non-prod ports `19180`/`19181`).  
5. `ZEROLLAMA_RUNTIME_URL=http://127.0.0.1:8081 ./scripts/e2e_coordination_smoke.sh` — mirrors + `go_training_gpu_busy` on `/health`.  
6. **5080 session + training ops:** `RUN_E2E_TRAINING_OPS=1 ./scripts/gpu_5080_session.sh` (serve must have `OLLAMA_TRAINING=true`).  
7. `OLLAMA_TRAINING=false` — training routes off.

---

## Known gaps / watch list

- **Auth:** `/api/train/*` and TCP `:9500` not authenticated in v1.  
- **`training.py` global `STATE`:** fragile if cached by value; refactor later.  
- **Go sched tests:** occasional flaky timeout in some envs — not training-specific.  
- **Reserve tuning:** 2 GiB default may be wrong for 12 GB vs 24 GB cards — measure, edit `admission.py`.

---

## Suggested next steps

1. Read [gpu-training.md](./gpu-training.md) + [phase11-runtime-admission.md](./phase11-runtime-admission.md).  
2. Run `e2e_training_ops_smoke.sh` and coordination smoke on GPU host.  
3. One real training job + concurrent `POST /api/chat` — confirm reserve and defer behavior.  
4. Tune `TRAINING_VRAM_RESERVE_BYTES` on target hardware (no new env var).

---

## This document

Context for teammates without the chat log. Protocol and knobs: [gpu-training.md](./gpu-training.md). Runtime admission: [phase11-runtime-admission.md](./phase11-runtime-admission.md).
