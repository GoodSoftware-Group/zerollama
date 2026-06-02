"""Opinionated inference-first policy driven by queue/VRAM metrics (Phase 11).

Why this module exists: Python cannot see ggml runners; Go pushes queue mirrors.
We throttle batch (low) work when the card is busy — without a dozen ADMISSION_* env vars.

Thresholds default in code (tune after measurement on target GPU, e.g. RTX 5080).
Optional env overrides (advanced ops — prefer measuring then setting once on serve).
Operators disable scheduling with ZEROLLAMA_RUNTIME_INFERENCE_POLICY=off only.
VRAM checks are separate (ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM).
"""

from __future__ import annotations

import os
import threading
from typing import Any

from runtime.gpu.priority import InferencePriority

def _policy_int(key: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


def _refresh_policy_thresholds() -> None:
    """Load thresholds from env (call at import; tests may reload module)."""
    global RUNTIME_BACKLOG_BATCH_MIN
    global DEFER_WAITING_MIN
    global GGML_SCHED_BACKLOG_MIN
    global CROSS_QUEUE_PRESSURE_ON
    global CROSS_QUEUE_PRESSURE_CLEAR
    global LOW_PRIORITY_VRAM_FACTOR
    # Aligned with Go ZEROLLAMA_GGML_PAUSE_RUNTIME_MIN_BACKLOG (default 4).
    RUNTIME_BACKLOG_BATCH_MIN = _policy_int(
        "ZEROLLAMA_RUNTIME_BACKLOG_BATCH_MIN", 4
    )
    DEFER_WAITING_MIN = _policy_int("ZEROLLAMA_RUNTIME_DEFER_WAITING_MIN", 1)
    GGML_SCHED_BACKLOG_MIN = _policy_int(
        "ZEROLLAMA_RUNTIME_GGML_SCHED_BACKLOG_MIN", 1
    )
    on = _policy_int("ZEROLLAMA_RUNTIME_CROSS_QUEUE_PRESSURE_ON", 6)
    clear = _policy_int("ZEROLLAMA_RUNTIME_CROSS_QUEUE_PRESSURE_CLEAR", 4, minimum=0)
    if clear >= on:
        clear = max(0, on - 1)
    CROSS_QUEUE_PRESSURE_ON = on
    CROSS_QUEUE_PRESSURE_CLEAR = clear
    raw_factor = os.environ.get(
        "ZEROLLAMA_RUNTIME_LOW_PRIORITY_VRAM_FACTOR", ""
    ).strip()
    if raw_factor:
        try:
            LOW_PRIORITY_VRAM_FACTOR = max(1.0, float(raw_factor))
        except ValueError:
            LOW_PRIORITY_VRAM_FACTOR = 1.5
    else:
        LOW_PRIORITY_VRAM_FACTOR = 1.5


# Defaults until _refresh_policy_thresholds() runs.
RUNTIME_BACKLOG_BATCH_MIN = 4
DEFER_WAITING_MIN = 1
GGML_SCHED_BACKLOG_MIN = 1
CROSS_QUEUE_PRESSURE_ON = 6
CROSS_QUEUE_PRESSURE_CLEAR = 4
LOW_PRIORITY_VRAM_FACTOR = 1.5
VRAM_MIN_FREE_DEFAULT_BYTES = 1024**3  # 1 GiB when GPU admission is on

_refresh_policy_thresholds()

_backpressure_latched = False
_backpressure_lock = threading.Lock()


def inference_first_enabled() -> bool:
    v = os.environ.get("ZEROLLAMA_RUNTIME_INFERENCE_POLICY", "inference-first").strip().lower()
    return v not in ("off", "0", "false", "no", "disabled")


def backpressure_snapshot(
    *,
    runtime_waiting: int,
    runtime_running: int,
) -> dict[str, Any]:
    from runtime.go_coordination import (
        cross_queue_depth,
        cross_queue_pressure_score,
        go_coordination_meta,
    )

    depth = cross_queue_depth(
        runtime_waiting=runtime_waiting, runtime_running=runtime_running
    )
    pressure = cross_queue_pressure_score(
        runtime_waiting=runtime_waiting, runtime_running=runtime_running
    )
    latched = _cross_queue_pressure_latched(
        runtime_waiting=runtime_waiting, runtime_running=runtime_running
    )
    return {
        "inference_first": inference_first_enabled(),
        "coordination": go_coordination_meta(),
        **depth,
        "cross_queue_pressure": pressure,
        "cross_queue_pressure_latched": latched,
        "thresholds": {
            "runtime_backlog_batch_min": RUNTIME_BACKLOG_BATCH_MIN,
            "defer_waiting_min": DEFER_WAITING_MIN,
            "ggml_sched_backlog_min": GGML_SCHED_BACKLOG_MIN,
            "cross_queue_pressure_on": CROSS_QUEUE_PRESSURE_ON,
            "cross_queue_pressure_clear": CROSS_QUEUE_PRESSURE_CLEAR,
        },
    }


def _cross_queue_pressure_latched(
    *,
    runtime_waiting: int,
    runtime_running: int,
) -> bool:
    """Hysteresis on combined queue pressure (fresh Go mirror required)."""
    from runtime.go_coordination import (
        cross_queue_pressure_score,
        go_coordination_is_fresh,
    )

    if not inference_first_enabled() or not go_coordination_is_fresh():
        return False
    score = cross_queue_pressure_score(
        runtime_waiting=runtime_waiting, runtime_running=runtime_running
    )
    global _backpressure_latched
    with _backpressure_lock:
        if _backpressure_latched:
            if score <= CROSS_QUEUE_PRESSURE_CLEAR:
                _backpressure_latched = False
            return _backpressure_latched
        if score >= CROSS_QUEUE_PRESSURE_ON:
            _backpressure_latched = True
        return _backpressure_latched


def batch_priority_subject_to_backpressure(priority: InferencePriority) -> bool:
    """Only batch/low priority is throttled at enqueue; normal chat keeps flowing.

    Why: default /api/chat is normal; generate_batch defaults to low in engine.
    """
    if not inference_first_enabled():
        return False
    return priority == InferencePriority.LOW


def cross_fifo_blocks_low(*, runtime_oldest_fifo: int) -> bool:
    """True when Go-side waiting work has an older global ticket than runtime queue head."""
    if not inference_first_enabled():
        return False
    from runtime.go_coordination import go_coordination_is_fresh, go_fifo_oldest_ggml

    if not go_coordination_is_fresh():
        return False
    go_oldest = go_fifo_oldest_ggml()
    if go_oldest == 0:
        return False
    if runtime_oldest_fifo <= 0:
        return True
    return go_oldest < runtime_oldest_fifo


def inference_backpressure_blocks_low(
    *,
    runtime_waiting: int,
    runtime_running: int,
    runtime_oldest_fifo: int = 0,
) -> bool:
    """True when inference-first metrics say batch (low) work should wait.

    Used at enqueue (reject LOW) and at dequeue (stall only when queue head is LOW).
    NORMAL chat is not blocked by runtime backlog or mirrored ggml/defer pressure.
    """
    if not inference_first_enabled():
        return False
    if cross_fifo_blocks_low(runtime_oldest_fifo=runtime_oldest_fifo):
        return True
    metrics = backpressure_snapshot(
        runtime_waiting=runtime_waiting, runtime_running=runtime_running
    )
    if int(metrics["runtime_backlog"]) >= RUNTIME_BACKLOG_BATCH_MIN:
        return True
    if not metrics.get("go_mirror_fresh"):
        return False
    if metrics.get("ggml_loads_paused"):
        return True
    if int(metrics.get("go_defer_waiting", 0)) >= DEFER_WAITING_MIN:
        return True
    if int(metrics.get("go_ggml_backlog", 0)) >= GGML_SCHED_BACKLOG_MIN:
        return True
    return bool(metrics.get("cross_queue_pressure_latched"))


def backpressure_rejection_reason(
    *,
    runtime_waiting: int,
    runtime_running: int,
    runtime_oldest_fifo: int = 0,
) -> str | None:
    """Human-readable trigger for metrics-driven reject (enqueue diagnostics)."""
    if cross_fifo_blocks_low(runtime_oldest_fifo=runtime_oldest_fifo):
        from runtime.go_coordination import go_fifo_oldest_ggml

        return (
            f"cross-queue FIFO (ggml ticket {go_fifo_oldest_ggml()} "
            f"< runtime {runtime_oldest_fifo})"
        )
    if not inference_backpressure_blocks_low(
        runtime_waiting=runtime_waiting,
        runtime_running=runtime_running,
        runtime_oldest_fifo=runtime_oldest_fifo,
    ):
        return None
    metrics = backpressure_snapshot(
        runtime_waiting=runtime_waiting, runtime_running=runtime_running
    )
    if int(metrics["runtime_backlog"]) >= RUNTIME_BACKLOG_BATCH_MIN:
        return f"runtime backlog {metrics['runtime_backlog']}"
    if metrics.get("ggml_loads_paused"):
        return "ggml loads paused"
    if int(metrics.get("go_defer_waiting", 0)) >= DEFER_WAITING_MIN:
        return f"defer waiting {metrics['go_defer_waiting']}"
    if int(metrics.get("go_ggml_backlog", 0)) >= GGML_SCHED_BACKLOG_MIN:
        return f"ggml backlog {metrics['go_ggml_backlog']}"
    if metrics.get("cross_queue_pressure_latched"):
        return f"cross-queue pressure {metrics['cross_queue_pressure']}"
    return "inference backpressure"
