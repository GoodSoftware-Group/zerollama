# Handoff: GPU training integration (Go + Python)

**Audience:** Another engineer with access to this repo who did not participate in the design and implementation thread.

**Purpose:** Capture **intent**, **decisions**, and **where to look next** so you can operate, extend, or review the work without replaying the full conversation.

---

## What we were solving

We needed **GPU fine-tuning / training** integrated into the Ollama daemon without:

- A second public server that clients had to discover separately, or  
- Python binding the same TCP port as external training clients.

**Decision:** **Go** is the only process that listens on the **public** training surfaces (HTTP and, by default, **TCP `:9500`**). **One Python process** (`trainingdaemon`) owns PyTorch/CUDA, listens only on a **private Unix domain socket**, and talks to Go via **gRPC + Protobuf** (`proto/training.proto`). Repo-root **`training.py`** remains the core training logic, imported as a library—not a standalone network server when Go is in charge.

**Why gRPC internally:** Strong typing, streaming (progress, OOM), and a clear evolution path compared to newline-delimited JSON between processes.

**Why inference-first VRAM on OOM (v1):** Single-GPU setups contend for VRAM. When training signals CUDA OOM, Go **pauses new inference loads**, **unloads inference runners**, then **acks** Python so `load_model` can retry once. Mid-training-loop OOM still fails the job (no fake checkpoint resume); we only notify Go so the next job sees a cleaner GPU—**why:** safe default.

---

## Conversation arc (compressed)

1. Started from “something on port 9500 like `training.py`” and language choices (Go/Rust/Python for GPU).  
2. Converged on **Python for training**, **Go for all public I/O** and scheduler integration.  
3. Internal wire format deliberately **not** the legacy JSON protocol—**Protobuf/gRPC over UDS**.  
4. **`OLLAMA_TRAINING`** defaults to **on** so the feature is visible; operators without the Python stack set `OLLAMA_TRAINING=false`.  
5. Follow-up work identified in review: double progress emit (fixed), TCP deadlines vs long sync jobs (fixed), pointless OOM wait on failed paths (fixed), graceful gRPC `stop` on shutdown, map cleanup after eviction, `defer ResumeLoads` in the OOM bridge.

---

## What to read first (technical)

| Doc | Role |
|-----|------|
| [gpu-training.md](./gpu-training.md) | Architecture, env vars, APIs, OOM bridge, troubleshooting, code map. |
| [ROADMAP.md § GPU training](./ROADMAP.md#gpu-training-fine-tuning) | Directional follow-ups and **non-goals**. |
| [CHANGELOG.md](../CHANGELOG.md) (Unreleased) | User-facing summary when this ships in a release. |
| [x/trainingdaemon/README.md](../x/trainingdaemon/README.md) | Python venv / `PYTHONPATH` for local dev. |
| [x/trainingworker/README.md](../x/trainingworker/README.md) | Why the Go bridge is a separate package. |

---

## Code map (quick)

| Layer | Path |
|--------|------|
| Proto | `proto/training.proto` |
| Go: spawn, gRPC, TCP :9500 | `x/trainingworker/client.go` |
| Go: generated stubs | `x/trainingworker/trainingpb/` |
| Go: HTTP `/api/train/*` | `server/training_api.go` |
| Go: wiring in `Serve()` | `server/routes.go` (training start, signal order, `ServePublicTCP` goroutine) |
| Go: scheduler hooks | `server/sched.go` — `PauseNewLoads`, `ResumeLoads`, `UnloadAllRunners`, `GetRunner` pause loop |
| Env | `envconfig/config.go` — `OLLAMA_TRAINING`, `OLLAMA_TRAINING_TCP`, `OLLAMA_TRAINING_PYTHONPATH` |
| Python: gRPC server | `x/trainingdaemon/trainingdaemon/server.py` |
| Python: bridge to `training.py` | `x/trainingdaemon/trainingdaemon/gpu_session.py` |
| Training logic | `training.py` |

---

## How to sanity-check locally

1. `python3` on `PATH`, deps installed per `x/trainingdaemon/pyproject.toml` (torch, grpcio, transformers, …).  
2. From repo: run the main server with training enabled (default). If the daemon is not found next to the binary, set **`OLLAMA_TRAINING_PYTHONPATH`** to the absolute path of `x/trainingdaemon` (the directory that **contains** the `trainingdaemon` package folder).  
3. **`GET /api/train/status`** (or `POST /api/train/jobs` with a minimal payload) if HTTP is up; **`echo '{"cmd":"ping"}' \| nc … 9500`** style checks for TCP (exact client left to you).  
4. Set **`OLLAMA_TRAINING=false`** to confirm training routes and subprocess stay off.

---

## Known gaps / watch list

- **`go test ./server`:** `TestList` has been observed failing in some environments (model count mismatch—likely environment / fixture pollution, not training-specific). Confirm on CI before assuming training broke it.  
- **Auth:** Public TCP and `/api/train/*` are not authenticated in v1; roadmap calls out adding policy when you harden deployments.  
- **`training.py` global `STATE`:** The daemon replaces `training.STATE` with a `BridgeState` subclass; it works because names resolve at call time—**fragile** if someone caches `STATE` by value; a later refactor can thread state explicitly.

---

## Suggested next steps for the next owner

1. Read [gpu-training.md](./gpu-training.md) end-to-end.  
2. Run one **async** job (`POST /api/train/jobs`) and one **legacy TCP** `ping` / `submit_job` if you care about backward compatibility.  
3. Decide whether to land remaining repo changes in one PR or split (proto + Go + Python + docs).  
4. If you add CI: start with **import/grpc smoke** (no full CUDA train in default PR path).

---

## This document

Added so a teammate can pick up **context and rationale** without the chat log. For protocol details and **why** each knob exists, prefer [gpu-training.md](./gpu-training.md).
