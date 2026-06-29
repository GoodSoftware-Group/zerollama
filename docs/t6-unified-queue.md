# T6 — Unified queue policy (operator guide)

**Status:** **Partial** — idle-wait, priority classes, defer queue (`defer-*`), allowed window, and cross-queue FIFO are shipped. Richer SLO classes and stricter inference/training separation remain roadmap.

This document is the **T6 operator runbook**. For VRAM broker + Phase 11–13 context, see [scheduling-vram-policy.md](./scheduling-vram-policy.md). For embedded training wiring, see [gpu-training.md](./gpu-training.md).

---

## Why T6 exists

Inference (chat → ggml and/or Python runtime) and training (`/api/train/*` → PyTorch) are **separate FIFOs** on purpose: training epochs are not safely preemptible from Go. T6 adds **coordination policy** so operators can:

1. **Reject or defer** training submit while inference is busy (idle-wait).
2. **Queue batch jobs** instead of spamming HTTP 409 (`defer-*` IDs).
3. **Restrict training to a time window** (night batch).
4. **Order work fairly** across ggml, runtime, and defer via global FIFO tickets.

**Smoke:** `./scripts/e2e_t6_queue_smoke.sh` (offline Go + pytest; optional live `/api/status` + runtime `/health`).

---

## Architecture (coordination layers)

```text
Training submit (POST /api/train/jobs or TCP :9500)
  │
  ├─ Allowed window gate? ──► 409 or defer
  ├─ Idle-wait gate?       ──► 409 or defer (ggml pending/active/loaded + runtime /health)
  ├─ Priority bypass?      ──► high/interactive skips idle-wait
  └─ Defer queue           ──► defer-<uuid> until idle + FIFO allows promote

Cross-queue FIFO (global tickets)
  Go AllocCrossQueueSeq() ──► ggml pending, defer enqueue, runtime requests
  Mirror POST /internal/go-coordination ──► runtime /health go_coordination
  Python batch/low waits when ggml ticket < runtime head; defer promote waits for inference
```

---

## Production checklist (single GPU)

```bash
export OLLAMA_MAX_LOADED_MODELS=1
export ZEROLLAMA_TRAINING_WAIT_INFERENCE_IDLE=1
export ZEROLLAMA_TRAINING_QUEUE_ON_BUSY=1          # queue instead of 409 when busy
export ZEROLLAMA_TRAINING_WAIT_GGML_LOADED=1       # default with idle-wait; loaded runner = busy
# Optional night batch:
# export ZEROLLAMA_TRAINING_ALLOWED_WINDOW=22:00-06:00
# export ZEROLLAMA_TRAINING_WINDOW_TZ=America/Los_Angeles
```

**Multi-GPU:** if ggml on GPU0 must not block training on GPU1, set `ZEROLLAMA_TRAINING_WAIT_GGML_LOADED=0`.

**Verify policy at runtime:**

```bash
curl -s http://127.0.0.1:8080/api/status | jq '.inference.training.queue_policy'
curl -s http://127.0.0.1:8081/health | jq '.go_coordination | with_entries(select(.key|startswith("fifo_")))'
```

---

## Priority matrix (training submit)

| `priority` | Idle-wait | Defer when busy | Typical use |
|------------|-----------|-----------------|-------------|
| `normal` (default) | **On** when env set | When `queue_on_busy` / env | Daytime fine-tune |
| `low` / `batch` | **On** | **Prefer defer** | Overnight / Wan video jobs |
| `high` / `interactive` | **Bypass** | N/A (direct submit) | Operator override |

HTTP body fields: `priority`, `queue_on_busy`. TCP training wire accepts the same keys.

---

## Idle-wait gate

**Env:** `ZEROLLAMA_TRAINING_WAIT_INFERENCE_IDLE=1`

Training submit is rejected (HTTP **409**) while:

- ggml scheduler has pending/active work, or (by default) **loaded** runners (`ZEROLLAMA_TRAINING_WAIT_GGML_LOADED`)
- Python runtime `/health` reports `waiting` / `running` / `llama_server: true`

**Fail-closed on probe errors:** `ZEROLLAMA_TRAINING_WAIT_FAIL_CLOSED` (default on with idle-wait) — if `/health` cannot be read, submit is rejected rather than assuming idle.

Code: `server/inference_workload.go`, guard in `x/trainingworker`.

---

## Defer queue (`defer-*` job IDs)

**Env:** `ZEROLLAMA_TRAINING_QUEUE_ON_BUSY=1` (or per-request `queue_on_busy: true`, or `priority: low`)

When policy would return **409**, Go accepts the job into an in-memory queue instead:

| Reject reason | Defer requires |
|---------------|----------------|
| Inference backlog | `ZEROLLAMA_TRAINING_WAIT_INFERENCE_IDLE=1` |
| Outside allowed window | `ZEROLLAMA_TRAINING_ALLOWED_WINDOW` set |
| Misconfigured window | **No defer** — HTTP **503**, fix env |

**Behavior:**

- Poll: `ZEROLLAMA_TRAINING_QUEUE_POLL_SECS` (default **5s**)
- Max depth: `ZEROLLAMA_TRAINING_QUEUE_MAX` (default **32**)
- Promotion returns real `job_id` via `promotedJobId` on the defer record
- Tombstone TTL: `ZEROLLAMA_TRAINING_QUEUE_TOMBSTONE_SECS` (default **24h**) — defer IDs stay queryable after promotion
- Retries: `ZEROLLAMA_TRAINING_QUEUE_RETRY_MAX` / `RETRY_SECS`
- Cancel: `DELETE /api/train/jobs/defer-…` (waiting only)
- List: merged into `GET /api/train/jobs` (waiting only unless `ZEROLLAMA_TRAINING_QUEUE_LIST_ALL=1`)

**Client note:** save `promotedJobId` before tombstone eviction if you poll only `defer-*`.

Code: `server/training_defer_queue.go`, `server/training_submit.go`.

---

## Allowed window (night batch)

**Env:** `ZEROLLAMA_TRAINING_ALLOWED_WINDOW=22:00-06:00` (HH:MM-HH:MM; spans midnight when start > end)

**Timezone:** `ZEROLLAMA_TRAINING_WINDOW_TZ` (IANA name or `local`, default local)

Outside window → **409** (or defer when `ZEROLLAMA_TRAINING_QUEUE_ON_BUSY=1`).

**Invalid window string** → all training submit blocked (**503**) until fixed — avoids silently running without the intended window.

Code: `server/training_window.go`, `envconfig/training_window.go`.

---

## Cross-queue FIFO

Go allocates monotonic tickets (`POST /internal/cross-queue-seq`, **loopback only**). Python calls Go via `ZEROLLAMA_GO_URL` (set automatically by `zerollama serve` from `OLLAMA_HOST`).

**Mirror fields** on runtime `/health` → `go_coordination`:

| Field | Meaning |
|-------|---------|
| `fifo_go_oldest_ggml` | Oldest ggml pending or in-flight load ticket |
| `fifo_go_oldest_defer` | Oldest waiting defer job ticket |
| `fifo_go_oldest_inference` | Min(ggml, mirrored runtime) inference head |
| `fifo_runtime_oldest` | Oldest runtime waiting **or running** ticket |

**Ordering rules:**

- Python `priority: low` / batch blocks when ggml ticket is older than runtime head.
- Ggml `pendingPopNext` yields while runtime FIFO is ahead.
- Defer promotion waits while inference (runtime or ggml) is ahead of the defer ticket.
- `useLoadedRunner` / model-prefer pop are **not** ticket-ordered.

**Stale mirror:** snapshots older than `ZEROLLAMA_RUNTIME_GO_COORDINATION_TTL_S` (default **30s**) are ignored for cross-queue admission (fail-open).

Code: `server/cross_queue_fifo.go`, `server/cross_fifo_policy.go`, `runtime/runtime/cross_queue_seq.py`, `runtime/runtime/go_coordination.py`.

---

## Environment reference (T6)

| Variable | Default | Purpose |
|----------|---------|---------|
| `ZEROLLAMA_TRAINING_WAIT_INFERENCE_IDLE` | off | Reject/defer training while inference busy |
| `ZEROLLAMA_TRAINING_WAIT_GGML_LOADED` | on when idle-wait | Treat loaded ggml runners as busy |
| `ZEROLLAMA_TRAINING_WAIT_FAIL_CLOSED` | on when idle-wait | Reject when runtime `/health` unreadable |
| `ZEROLLAMA_TRAINING_QUEUE_ON_BUSY` | off | Defer instead of 409 |
| `ZEROLLAMA_TRAINING_QUEUE_POLL_SECS` | 5 | Defer drain interval |
| `ZEROLLAMA_TRAINING_QUEUE_MAX` | 32 | Max waiting defer jobs (0 = unlimited) |
| `ZEROLLAMA_TRAINING_QUEUE_TOMBSTONE_SECS` | 86400 | How long terminal defer records stay queryable |
| `ZEROLLAMA_TRAINING_QUEUE_RETRY_MAX` | 3 | Promotion retry count |
| `ZEROLLAMA_TRAINING_QUEUE_RETRY_SECS` | 60 | Min wait between retries |
| `ZEROLLAMA_TRAINING_QUEUE_LIST_ALL` | off | Include terminal defer jobs in list API |
| `ZEROLLAMA_TRAINING_ALLOWED_WINDOW` | unset | HH:MM-HH:MM training window |
| `ZEROLLAMA_TRAINING_WINDOW_TZ` | local | Window timezone |
| `ZEROLLAMA_RUNTIME_GO_COORDINATION_TTL_S` | 30 | Stale mirror threshold |
| `ZEROLLAMA_GO_URL` | auto from serve | Python → Go internal URL |

---

## GET /api/status — `inference.training.queue_policy`

Fleet and operators can poll configured T6 gates without parsing env:

```json
{
  "inference": {
    "training": {
      "queue_policy": {
        "wait_inference_idle": true,
        "wait_ggml_loaded": true,
        "wait_fail_closed": true,
        "queue_on_busy": true,
        "allowed_window": "22:00-06:00",
        "allowed_window_enabled": true,
        "cross_queue_fifo": true,
        "defer_waiting": 0,
        "defer_tracked": 0
      }
    }
  }
}
```

---

## Code map

| Area | Path |
|------|------|
| Training submit + defer | `server/training_submit.go`, `server/training_defer_queue.go` |
| Idle-wait gate | `server/inference_workload.go` |
| Allowed window | `server/training_window.go`, `envconfig/training_window.go` |
| Cross-queue FIFO | `server/cross_queue_fifo.go`, `server/cross_fifo_policy.go` |
| Coordination push | `server/training_policy.go`, `runtime/runtime/go_coordination.py` |
| Runtime defer admission | `runtime/runtime/gpu/admission.py` |
| Env helpers | `envconfig/config.go` |
| Smoke | `scripts/e2e_t6_queue_smoke.sh` |

---

## What's next (roadmap)

- Richer **SLO classes** (latency tiers beyond normal/low/high).
- Stricter **training vs inference class separation** on shared schedulers.
- Deeper load-test script under real chat + training contention (complements Phase 11).

See [ROADMAP.md — T6](./ROADMAP.md#phases-training-track).
