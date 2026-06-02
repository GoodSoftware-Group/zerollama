"""Native decode-step counter (Phase 15 v6 — hot-path hook).

Counts ``llama_decode`` (and encoder ``llama_encode``) invocations on the in-process
ctypes path. Full token generation in C is not implemented; this is scaffolding for
a future native decode loop wired to block tables.
"""

from __future__ import annotations

import os
import threading

_py_decode_steps = 0
_py_lock = threading.Lock()


def decode_hook_enabled() -> bool:
    raw = os.environ.get("ZEROLLAMA_RUNTIME_KV_DECODE_HOOK", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    return True


def native_decode_available() -> bool:
    try:
        from runtime.kv._kv_native import decode_step  # noqa: F401

        return True
    except ImportError:
        return False
    except Exception:
        return False


def current_decode_steps() -> int:
    """Cumulative decode steps (native or Python fallback)."""
    if native_decode_available():
        from runtime.kv._kv_native import decode_step

        return int(decode_step(0))
    with _py_lock:
        return _py_decode_steps


def decode_steps_delta(before: int, after: int) -> int:
    return max(0, after - before)


def record_decode_step(steps: int = 1) -> int | None:
    """Record decode work; return cumulative steps or None if hook disabled."""
    if not decode_hook_enabled() or steps <= 0:
        return None
    global _py_decode_steps
    if native_decode_available():
        from runtime.kv._kv_native import decode_step

        return int(decode_step(int(steps)))
    with _py_lock:
        _py_decode_steps += int(steps)
        return _py_decode_steps


def decode_steps_health(*, llama_backend: str) -> dict[str, int | str | bool] | None:
    if not decode_hook_enabled():
        return {
            "active": False,
            "reason": "ZEROLLAMA_RUNTIME_KV_DECODE_HOOK=0",
        }
    if llama_backend != "inprocess":
        return {
            "active": False,
            "reason": (
                f"backend={llama_backend} (decode hook runs on in-process ctypes path only)"
            ),
        }
    native = native_decode_available()
    return {
        "active": True,
        "value": current_decode_steps(),
        "source": "native" if native else "python",
        "hook": "llama_decode",
    }


def reset_decode_steps_for_tests() -> None:
    global _py_decode_steps
    with _py_lock:
        _py_decode_steps = 0
