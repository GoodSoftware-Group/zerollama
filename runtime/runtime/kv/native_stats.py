"""Read-only native KV counters (Phase 15 v7)."""

from __future__ import annotations

from typing import Any


def native_kv_available() -> bool:
    try:
        from runtime.kv._kv_native import kv_stats  # noqa: F401

        return True
    except ImportError:
        return False
    except Exception:
        return False


def native_kv_stats() -> dict[str, int] | None:
    if not native_kv_available():
        return None
    from runtime.kv._kv_native import kv_stats

    raw = kv_stats()
    return {
        "scheduler_tick": int(raw["scheduler_tick"]),
        "decode_steps": int(raw["decode_steps"]),
    }
