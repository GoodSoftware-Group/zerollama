# Phase 11 — Python runtime admission & VRAM policy

**Audience:** Operators and contributors working on single-GPU (or tight VRAM) hosts where **Go ggml runners**, the **Python runtime** (`llama-server`), and **embedded training** share one card.

**Related:** [scheduling-vram-policy.md](./scheduling-vram-policy.md) (full stack), [runtime/docs/OPERATIONS.md](../runtime/docs/OPERATIONS.md) (`/health` fields), [handoff-phase12-runtime-tools.md](./handoff-phase12-runtime-tools.md) (tools + GPU thread).

---

## Why Phase 11 exists

On a consumer GPU, three things compete for the same VRAM and attention:

1. **Interactive inference** (chat, agents) — latency-sensitive.
2. **Batch / background inference** (many prompts, `generate_batch`) — throughput-oriented.
3. **Training** (fine-tuning) — long-running, hard to preempt safely mid-epoch.

Zerollama already had:

- **Go VRAM broker (Phase 8)** — proactive unload before loads.
- **Separate FIFOs** for ggml, runtime, and training — not one global optimizer.

Phase 11 adds **Python-side policy** so the runtime does not accept work it cannot run **before** KV allocation and `llama-server` start. **Why before load:** subprocess OOM and queue buildup are slow, opaque, and hard to recover from on a full 16 GB card.

**What we deliberately did not do:** dozens of `ADMISSION_*` env toggles for every threshold. Defaults live in code (`admission.py`, `inference_policy.py`); after measurement on a target GPU, either commit new defaults or set the **optional** `ZEROLLAMA_RUNTIME_*` backpressure overrides on **serve** (see table below). VRAM headroom uses env (`VRAM_MIN_FREE`, `TRAINING_VRAM_RESERVE`) — not the same layer as backlog thresholds.

---

## Operator surface (two policy knobs)

| Variable | Default | When you set it |
|----------|---------|-----------------|
| `ZEROLLAMA_RUNTIME_INFERENCE_POLICY` | `inference-first` | `off` — disable defer/ggml/backlog **throttling for `priority: low` only** |
| `ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM` | auto (on if probe works) | `0` — disable host + GPU budget checks and 1 GiB headroom floor |
| `ZEROLLAMA_RUNTIME_VRAM_MIN_FREE` | `1GiB` | Minimum free VRAM when checks are on (size string: `512MiB`, `1GiB`, …; prefer **GiB** over `GB` — `GB` uses decimal 1000-based bytes) |
| `ZEROLLAMA_RUNTIME_TRAINING_VRAM_RESERVE` | `2GiB` | Headroom subtracted while training holds the GPU (handoff / pause / Go busy) |

**Optional safety cap (not policy):** `ZEROLLAMA_RUNTIME_MAX_QUEUE` (default **512** in code). **Why separate:** prevents unbounded memory from queued KV block tables; not an admission tuning knob.

**Estimate tuning (Phase 13, not Phase 11):** `ZEROLLAMA_RUNTIME_VRAM_MARGIN`, `VRAM_ESTIMATE_FACTOR`, autotune, probe calibrate, `VRAM_CLAMP_NUM_CTX` — calibration on real weights/KV, not “who gets the GPU when busy.” See [phase13-runtime-vram.md](./phase13-runtime-vram.md).

---

## VRAM headroom (env or code defaults)

Defaults in `runtime/runtime/gpu/admission.py` — override on the host after measurement (e.g. RTX 5080):

| Setting | Default | Env |
|---------|---------|-----|
| Training reserve | 2 GiB | `ZEROLLAMA_RUNTIME_TRAINING_VRAM_RESERVE` |
| Min free VRAM | 1 GiB | `ZEROLLAMA_RUNTIME_VRAM_MIN_FREE` |

`/health` → `admission.vram_min_free_configured`, `admission.vram_training_reserve_configured`.

## Product constants (tune after measurement on 5080)

Defaults in `runtime/runtime/gpu/inference_policy.py`. Override on **serve** after `./scripts/gpu_phase13_snapshot.sh` + coordination smoke (advanced — prefer code defaults until measured):

| Constant | Default | Env override |
|----------|---------|----------------|
| `LOW_PRIORITY_VRAM_FACTOR` | 1.5× | `ZEROLLAMA_RUNTIME_LOW_PRIORITY_VRAM_FACTOR` |
| `RUNTIME_BACKLOG_BATCH_MIN` | 4 | `ZEROLLAMA_RUNTIME_BACKLOG_BATCH_MIN` |
| `DEFER_WAITING_MIN` | 1 | `ZEROLLAMA_RUNTIME_DEFER_WAITING_MIN` |
| `GGML_SCHED_BACKLOG_MIN` | 1 | `ZEROLLAMA_RUNTIME_GGML_SCHED_BACKLOG_MIN` |
| `CROSS_QUEUE_PRESSURE_ON` / `CLEAR` | 6 / 4 | `ZEROLLAMA_RUNTIME_CROSS_QUEUE_PRESSURE_ON`, `_CLEAR` |

**5080 session:** `./scripts/gpu_5080_session.sh` — preflight + `gpu_smoke_all` + snapshot JSON + `gpu_snapshot` hints. **Why:** proves admission fits on a 16GB smoke path; does **not** by itself retune backlog thresholds (need load under real chat+training). Guide: [gpu-5080-operator-guide.md](./gpu-5080-operator-guide.md).

**YAML defaults:** When autoconfig picks `single_gpu.yaml`, the `vram:` block sets min-free / training-reserve / autotune if env is unset — same semantics as the env table above; **why:** one place for 16GB installs. See [phase13-runtime-vram.md](./phase13-runtime-vram.md#why-vram-in-single_gpuyaml).

---

## Request priority (`options.priority`)

| Value | Aliases | Behavior |
|-------|---------|----------|
| `high` | `interactive`, `urgent` | Front of queue; **bypasses 1 GiB min-free gate**; still must fit **model estimate + training reserve** at load |
| `normal` | (default) | Standard VRAM checks; **not** blocked by defer/ggml/backlog mirrors |
| `low` | `batch`, `background` | 1.5× min free; **throttled** under inference-first; `generate_batch` defaults here |

**Why three classes:** aligns with Go training T6 `priority` without merging training and chat into one FIFO.

---

## What runs when (submit → queue → load)

```text
POST /api/generate|chat (or engine admit)
│
├─ Training handoff / pause?     → reject ALL (inference paused for training)
├─ Queue ≥ MAX_QUEUE?            → reject ALL
├─ Inference-first (LOW only)    → reject LOW if defer/ggml/backlog/pressure/FIFO
├─ Generic 1 GiB gate            → only if GGUF path unknown yet (bottleneck probe)
└─ Enqueue GGUF precheck         → host RAM + check_gguf_vram_budget (if path known)

Scheduler tick (dequeue)
│
├─ Repeat policy + GGUF precheck (VRAM may have dropped while waiting)
└─ Pop head:
      HIGH  → always if at front
      LOW   → stall if inference-first or cross-FIFO says wait
      NORMAL→ runs (not stalled by mirror gates)
```

**Why enqueue + dequeue both check VRAM:** free memory can change while requests wait (training started, another process, ggml loaded a runner). **Why skip duplicate 1 GiB gate when GGUF known:** `check_gguf_vram_budget` already applies `max(model×margin, min_free×priority)`.

**Why skippable precheck when same model+ctx loaded:** avoid re-probing every token of a streaming session; dequeue still re-checks when ctx grows or model swaps.

---

## Inference-first vs VRAM checks

These are **independent**:

- `INFERENCE_POLICY=off` → batch/low no longer rejected or stalled for Go mirror signals; **VRAM checks unchanged** if `CHECK_GPU_VRAM` is on.
- `CHECK_GPU_VRAM=0` → no host/GPU budget; inference-first can still throttle **low** if policy is on.

**Why:** “turn off scheduling policy” must not silently disable safety rails, and “disable VRAM probes” must not disable defer/ggml coordination.

---

## Go coordination (why Python needs mirrors)

Python cannot see ggml runner processes. Go pushes `POST /internal/go-coordination` (~400 ms with training monitor or coordination pusher). Python uses a **fresh** mirror (TTL default 30 s) for:

- Rejecting **low** at enqueue when defer/ggml busy.
- Stalling **low** at dequeue when queue head is low.

**Stale mirror → fail-open** on mirrored metrics. **Why:** better to admit batch work than deadlock chat when Go push fails; local runtime backlog gates still apply for **low**.

**Training reserve (authoritative):** `POST /internal/training-gpu-busy` from `server/training_policy.go` when training occupies GPU — not inferred from mirror alone.

---

## `/health` admission fields

| Field | Meaning |
|-------|---------|
| `gates_active.*` | Signal on (**does not mean all traffic blocked**). See names below. |
| `low_would_wait` | Metrics say **low** should wait (enqueue reject + dequeue stall if head is low) |
| `runtime_backlog_pressure` | `waiting+running ≥ 4` — **does not block normal** |
| `defer_would_block_low`, `ggml_*_would_block_low` | Mirror signals for **low** only |
| `gates_active_compat` | Legacy key names (`batch_backpressure`, …) for old dashboards |
| `vram_gate` | Follows `CHECK_GPU_VRAM` |
| `vram_training_reserve` | Bytes reserved in policy math when training busy |

After deploy, smoke: `./scripts/e2e_coordination_smoke.sh` (expects `low_would_wait` in `gates_active`).

---

## Code map

| Concern | Path |
|---------|------|
| Admission API | `runtime/runtime/gpu/admission.py` |
| Thresholds + backpressure | `runtime/runtime/gpu/inference_policy.py` |
| Priority parsing | `runtime/runtime/gpu/priority.py` |
| VRAM probe + budget | `runtime/runtime/gpu_vram.py` |
| Enqueue/dequeue orchestration | `runtime/runtime/engine.py` (`_vram_precheck_*`, `_check_admit_policy`) |
| Coordinator + `/health` snapshot | `runtime/runtime/gpu/mutex.py` |
| Dequeue pop policy | `runtime/runtime/scheduler/scheduler.py`, `loop.py` |
| Go mirror | `runtime/runtime/go_coordination.py` |
| Go training busy | `server/training_policy.go` → `/internal/training-gpu-busy` |

---

## Tests & smoke

```bash
cd runtime && PYTHONPATH=. python3 -m pytest tests/ -q --ignore=tests/test_supervisor.py
```

Notable tests: `test_admission.py`, `test_inference_policy.py`, `test_scheduler_low_dequeue.py`, `test_admit_vram_precheck.py`, `test_gpu_vram.py`.

```bash
ZEROLLAMA_RUNTIME_URL=http://127.0.0.1:8081 ./scripts/e2e_coordination_smoke.sh
```

---

## What remains (Phase 11 partial)

- Tune `TRAINING_VRAM_RESERVE_BYTES` and thresholds on target hardware (e.g. RTX 5080).
- Heuristic estimates can still false-reject/accept — use Phase 13 autotune/calibration, not more admission flags.
- Phase 14 (in-process llama) reduces loopback overhead; admission logic stays valid.
