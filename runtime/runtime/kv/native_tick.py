"""Scheduler admission tick counter (Phase 15 v4/v5).

Uses native ``scheduler_tick()`` when ``runtime.kv._kv_native`` is built; otherwise
a process-local Python counter so ``/health`` always exposes monotonic progress.
"""

from __future__ import annotations

import threading

_py_scheduler_tick = 0
_py_lock = threading.Lock()


def native_scheduler_available() -> bool:
    try:
        from runtime.kv._kv_native import scheduler_tick  # noqa: F401

        return True
    except ImportError:
        return False
    except Exception:
        return False


def record_scheduler_tick() -> int:
    """Increment tick counter; return new value."""
    global _py_scheduler_tick
    if native_scheduler_available():
        from runtime.kv._kv_native import scheduler_tick

        return int(scheduler_tick())
    with _py_lock:
        _py_scheduler_tick += 1
        return _py_scheduler_tick


def scheduler_tick_health(last_tick: int | None) -> dict[str, int | str]:
    """Shape for ``/health``."""
    native = native_scheduler_available()
    if last_tick is not None:
        out: dict[str, int | str] = {
            "value": int(last_tick),
            "source": "native" if native else "python",
        }
    else:
        with _py_lock:
            py_val = _py_scheduler_tick
        out = {
            "value": py_val if not native else 0,
            "source": "native" if native else "python",
        }
        out["note"] = "increments when SchedulerLoop.tick admits requests"
    if native and last_tick is not None:
        out["legacy_field"] = "kv_native_scheduler_tick matches value on last admit"
    return out


def reset_scheduler_tick_for_tests() -> None:
    """Test helper."""
    global _py_scheduler_tick
    with _py_lock:
        _py_scheduler_tick = 0
