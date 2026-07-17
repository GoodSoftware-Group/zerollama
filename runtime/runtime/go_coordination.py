"""Snapshots pushed from the Go daemon (training defer, policy flags)."""

from __future__ import annotations

import os
import threading
import time
from typing import Any

_lock = threading.Lock()
_snapshot: dict[str, Any] = {}
_updated_at: float | None = None


def _coordination_ttl_s() -> float:
    raw = os.environ.get("ZEROLLAMA_RUNTIME_GO_COORDINATION_TTL_S", "30").strip()
    try:
        return max(0.01, float(raw))
    except ValueError:
        return 30.0


def update_go_coordination(data: dict[str, Any] | None) -> None:
    global _snapshot, _updated_at
    with _lock:
        _snapshot = dict(data) if data else {}
        _updated_at = time.monotonic() if data else None


def go_coordination_is_fresh() -> bool:
    """False when Go has never pushed or the mirror is older than TTL."""
    with _lock:
        if _updated_at is None:
            return False
        return (time.monotonic() - _updated_at) <= _coordination_ttl_s()


def go_coordination_meta() -> dict[str, Any]:
    """Freshness metadata for /health and admission (fail-open when stale)."""
    with _lock:
        if _updated_at is None:
            return {"fresh": False, "stale": True, "age_s": None, "ttl_s": _coordination_ttl_s()}
        age = time.monotonic() - _updated_at
        fresh = age <= _coordination_ttl_s()
        return {
            "fresh": fresh,
            "stale": not fresh,
            "age_s": round(age, 3),
            "ttl_s": _coordination_ttl_s(),
        }


def go_coordination_health() -> dict[str, Any]:
    """Snapshot fields plus freshness metadata."""
    with _lock:
        out = dict(_snapshot)
        if _updated_at is None:
            meta = {
                "fresh": False,
                "stale": True,
                "age_s": None,
                "ttl_s": _coordination_ttl_s(),
            }
        else:
            age = time.monotonic() - _updated_at
            fresh = age <= _coordination_ttl_s()
            meta = {
                "fresh": fresh,
                "stale": not fresh,
                "age_s": round(age, 3),
                "ttl_s": _coordination_ttl_s(),
            }
    out["coordination"] = meta
    return out


def _defer_waiting_raw() -> int:
    raw = _snapshot.get("defer_waiting")
    if raw is None:
        return 0
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def go_defer_waiting() -> int:
    """Training jobs in Go defer queue (0 if mirror missing or stale)."""
    if not go_coordination_is_fresh():
        return 0
    with _lock:
        return _defer_waiting_raw()


def go_training_gpu_blocked() -> bool:
    """Go /health mirror only; VRAM reserve uses POST /internal/training-gpu-busy."""
    if not go_coordination_is_fresh():
        return False
    with _lock:
        return bool(_snapshot.get("training_gpu_blocked"))


def go_lmcache_blob_peers() -> list[str]:
    """L3-R11: peer bases pushed from Go (FLEET_PEERS) for HTTP blob pull."""
    if not go_coordination_is_fresh():
        return []
    with _lock:
        raw = _snapshot.get("lmcache_blob_peers")
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = raw.split(",")
    elif isinstance(raw, (list, tuple)):
        parts = list(raw)
    else:
        return []
    out: list[str] = []
    for part in parts:
        p = str(part).strip().rstrip("/")
        if p:
            out.append(p)
    return out


def _int_snapshot_field(key: str) -> int:
    raw = _snapshot.get(key)
    if raw is None:
        return 0
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def go_sched_pending() -> int:
    """ggml scheduler pending requests (0 if mirror stale)."""
    if not go_coordination_is_fresh():
        return 0
    with _lock:
        return _int_snapshot_field("sched_pending")


def go_sched_active() -> int:
    """ggml scheduler active runner refs (0 if mirror stale)."""
    if not go_coordination_is_fresh():
        return 0
    with _lock:
        return _int_snapshot_field("sched_active")


def go_sched_loaded() -> int:
    """ggml loaded runner count (0 if mirror stale)."""
    if not go_coordination_is_fresh():
        return 0
    with _lock:
        return _int_snapshot_field("sched_loaded")


def go_ggml_sched_backlog(*, include_loaded: bool = False) -> int:
    """Pending + active (+ optional loaded runners); 0 if mirror stale."""
    if not go_coordination_is_fresh():
        return 0
    with _lock:
        total = _int_snapshot_field("sched_pending") + _int_snapshot_field("sched_active")
        if include_loaded:
            total += _int_snapshot_field("sched_loaded")
        return total


def go_ggml_loads_paused() -> bool:
    """True when Go has paused new ggml loads (0/false if mirror stale)."""
    if not go_coordination_is_fresh():
        return False
    with _lock:
        return bool(_snapshot.get("ggml_loads_paused"))


def go_runtime_mirror_backlog() -> int:
    """runtime_waiting + runtime_running from Go /health probe; 0 if stale or absent."""
    if not go_coordination_is_fresh():
        return 0
    with _lock:
        return _int_snapshot_field("runtime_waiting") + _int_snapshot_field("runtime_running")


def _uint_snapshot_field(key: str) -> int:
    raw = _snapshot.get(key)
    if raw is None:
        return 0
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def go_fifo_oldest_ggml() -> int:
    if not go_coordination_is_fresh():
        return 0
    with _lock:
        return _uint_snapshot_field("fifo_go_oldest_ggml")


def go_fifo_oldest_defer() -> int:
    if not go_coordination_is_fresh():
        return 0
    with _lock:
        return _uint_snapshot_field("fifo_go_oldest_defer")


def go_fifo_oldest() -> int:
    """Smallest FIFO ticket among Go-side waiting work (ggml pending + defer); 0 if none."""
    if not go_coordination_is_fresh():
        return 0
    with _lock:
        explicit = _uint_snapshot_field("fifo_go_oldest")
        if explicit > 0:
            return explicit
        return min_nonzero(
            _uint_snapshot_field("fifo_go_oldest_ggml"),
            _uint_snapshot_field("fifo_go_oldest_defer"),
        )


def min_nonzero(a: int, b: int) -> int:
    if a <= 0:
        return b
    if b <= 0:
        return a
    return min(a, b)


def cross_queue_depth(*, runtime_waiting: int, runtime_running: int) -> dict[str, int | bool]:
    """Operator snapshot: local + mirrored queue pressure and FIFO head tickets."""
    mirror_fresh = go_coordination_is_fresh()
    out: dict[str, int | bool] = {
        "runtime_backlog": max(0, runtime_waiting) + max(0, runtime_running),
        "go_runtime_mirror": go_runtime_mirror_backlog(),
        "go_defer_waiting": go_defer_waiting(),
        "go_ggml_backlog": go_ggml_sched_backlog(include_loaded=False),
        "go_ggml_loaded": go_sched_loaded(),
        "ggml_loads_paused": go_ggml_loads_paused(),
        "go_mirror_fresh": mirror_fresh,
        "fifo_go_oldest": go_fifo_oldest(),
        "fifo_go_oldest_ggml": go_fifo_oldest_ggml(),
        "fifo_go_oldest_defer": go_fifo_oldest_defer(),
    }
    if not mirror_fresh:
        out["pressure_note"] = (
            "go_* counts are 0 when mirror stale; cross_queue_pressure may undercount"
        )
    return out


def cross_queue_pressure_score(*, runtime_waiting: int, runtime_running: int) -> int:
    """Single scalar for dashboards (not a scheduler — observability toward T6)."""
    d = cross_queue_depth(
        runtime_waiting=runtime_waiting, runtime_running=runtime_running
    )
    return int(d["runtime_backlog"]) + int(d["go_defer_waiting"]) + int(d["go_ggml_backlog"])
