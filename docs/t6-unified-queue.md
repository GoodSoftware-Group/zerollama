# T6 — Unified queue policy (operator guide)

**Status:** **Partial** — idle-wait, priority classes, defer queue (`defer-*`), allowed window, and cross-queue FIFO are shipped. **T6a:** unknown `priority` / `slo_class` → **400**; synonyms + aliases. **T6b:** submit/defer echo resolved `priority`. **T6c:** defer waiting pauses *new* ggml loads (default follows `queue_on_busy`) so continuous chat cannot starve promotion.

This document is the **T6 operator runbook**. For VRAM broker + Phase 11–13 context, see [scheduling-vram-policy.md](./scheduling-vram-policy.md). For embedded training wiring, see [gpu-training.md](./gpu-training.md).

---

## Why T6 exists

Inference (chat → ggml and/or Python runtime) and training (`/api/train/*` → PyTorch) are **separate FIFOs** on purpose: training epochs are not safely preemptible from Go. T6 adds **coordination policy** so operators can:

1. **Reject or defer** training submit while inference is busy (idle-wait).
2. **Queue batch jobs** instead of spamming HTTP 409 (`defer-*` IDs).
3. **Restrict training to a time window** (night batch).
4. **Order work fairly** across ggml, runtime, and defer via global FIFO tickets.

**Smoke:** `./scripts/e2e/e2e_t6_queue_smoke.sh` (offline Go + pytest; optional live `/api/status` + runtime `/health`).

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
# T6c: with queue_on_busy, new ggml loads pause while defer-* waits (starve-proof).
# Override: ZEROLLAMA_TRAINING_DEFER_PAUSES_GGML=0
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

| `priority` / `slo_class` | Idle-wait | Defer when busy | Typical use |
|--------------------------|-----------|-----------------|-------------|
| `normal` (default) | **On** when env set | When `queue_on_busy` / env | Daytime fine-tune |
| `low` / `batch` / `overnight` | **On** | **Prefer defer** | Night batch / Wan video jobs |
| `high` / `interactive` / `realtime` | **Bypass** | N/A (direct submit) | Operator override |

HTTP body fields: `priority`, `slo` / `slo_class` (synonym), `queue_on_busy`. TCP training wire accepts the same keys.

**WHY reject unknown labels:** silent coerce to `normal` hid typos (`higgh`) the same way unknown train `kind` used to hide `grpoo` → SFT. If both `priority` and `slo_class` are set, they must resolve to the same tier or the API returns **400**.

**T6b echo:** `POST /api/train/jobs` returns `"priority": "low|normal|high"` (resolved). Deferred job status / list entries include the same field so automation can assert `slo_class: overnight` → `priority: low` without re-parsing aliases.

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
| `ZEROLLAMA_TRAINING_DEFER_PAUSES_GGML` | = queue_on_busy | Pause **new** ggml loads while defer waiting (T6c) |
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
        "defer_pauses_ggml": true,
        "block_inference_during_training": true,
        "allowed_window": "22:00-06:00",
        "allowed_window_enabled": true,
        "cross_queue_fifo": true,
        "defer_waiting": 0,
        "defer_tracked": 0,
        "training_gpu_busy": false,
        "ggml_loads_paused": false
      }
    }
  }
}
```

### T6c — defer must not starve under continuous chat

**Problem:** idle-wait + `queue_on_busy` accepts `defer-*`, but new ggml loads / keep-alive can keep the GPU “busy” so promote never fires.

**Fix:** when `defer_waiting > 0` and `ZEROLLAMA_TRAINING_DEFER_PAUSES_GGML` is on (default = `queue_on_busy`), Go **pauses new ggml loads** only. In-flight generate finishes; no forced unload of active chats. Set `=0` if multi-GPU inference must keep loading while training waits on another device.

**Latency:** enqueue/cancel/promote call `syncRuntimeCoordinationAndPauses` immediately (not only the ~400ms training policy ticker), so `PauseNewLoads` / `ResumeLoads` track the defer depth without a beat of lag.

**Host unit:** `./scripts/e2e/e2e_t6_contention_unit.sh`

### T6d — HTTP contention path (busy ggml → defer → pause)

**Coverage:** `TestTrainHTTPDeferPausesGgmlLoads` posts `/api/train/jobs` while the ggml pending queue is non-empty (`queue_on_busy`), asserts `PauseNewLoads`, then `DELETE` cancels and resumes. Smoke: `./scripts/e2e/e2e_t6_contention_live.sh` (optional `RUN_E2E_T6_LIVE=1 OLLAMA_HOST=http://127.0.0.1:11435` probes lab `/api/status` only — never `:11434`).

---

## Code map

| Area | Path |
|------|------|
| Training submit + defer | `server/training_submit.go`, `server/training_defer_queue.go` |
| Idle-wait gate | `server/inference_workload.go` |
| Defer → pause ggml (T6c) | `server/inference_load_policy.go`, `envconfig.TrainingDeferPausesGgml` |
| Allowed window | `server/training_window.go`, `envconfig/training_window.go` |
| Cross-queue FIFO | `server/cross_queue_fifo.go`, `server/cross_fifo_policy.go` |
| Priority / SLO | `x/trainprio`, `server/training_priority.go` |
| Coordination push | `server/training_policy.go`, `runtime/runtime/go_coordination.py` |
| Runtime defer admission | `runtime/runtime/gpu/admission.py` |
| Env helpers | `envconfig/config.go` |
| Smoke | `scripts/e2e/e2e_t6_queue_smoke.sh`, `e2e_t6_contention_live.sh` |

---

## What's next (roadmap)

- ~~Wire-safe SLO synonyms + reject unknown priority~~ (**T6a**)
- ~~Echo resolved priority on submit/defer~~ (**T6b**)
- ~~Defer waiting pauses new ggml loads~~ (**T6c** + immediate sync on enqueue/cancel/promote)
- ~~HTTP busy→defer→pause path~~ (**T6d** — `e2e_t6_contention_live.sh`)
- Optional fourth latency tier only if operators need it — do **not** invent a second queue.

See [ROADMAP.md — T6](./ROADMAP.md#phases-training-track).
