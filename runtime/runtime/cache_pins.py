"""In-process L3 cache-pin registry (Go /api/cache/pin → /internal/cache/pin).

WHY: Hermes wants disk slot blobs for a prompt_cache_key to survive the default
``ZEROLLAMA_LLAMA_CACHE_TTL_MS`` while a lease is active. Go owns the public
lease API; Python only extends eviction horizons for derived slot ids.

Limitation: does not force idle llama-server id_slot retention — only disk TTL.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

_SLOT_BIN_RE = re.compile(r"^slot_(\d+)_\d+(?:\.(?:short|long|extended))?\.bin$")

# Extended horizon while pinned (7d) — bump relative to default 1h slot TTL.
PINNED_SLOT_TTL_MS = 7 * 24 * 60 * 60 * 1000


@dataclass
class _Pin:
    pin_id: str
    prompt_cache_key: str
    expires_at_ms: float
    slot_ids: set[int]


_lock = threading.Lock()
_pins: dict[str, _Pin] = {}


def _parse_expires_at(raw: str | None) -> float:
    if not raw:
        return time.time() * 1000 + PINNED_SLOT_TTL_MS
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.timestamp() * 1000
    except ValueError:
        return time.time() * 1000 + PINNED_SLOT_TTL_MS


def _expire_locked(now_ms: float) -> None:
    dead = [pid for pid, p in _pins.items() if p.expires_at_ms <= now_ms]
    for pid in dead:
        del _pins[pid]


def register_cache_pin(
    *,
    pin_id: str,
    prompt_cache_key: str,
    expires_at: str | None = None,
    parallel: int | None = None,
) -> dict[str, Any]:
    from runtime.cache_bridge import derive_slot_id

    key = (prompt_cache_key or "").strip()
    pid = (pin_id or "").strip()
    if not key or not pid:
        raise ValueError("pin_id and prompt_cache_key required")
    n = int(parallel) if parallel and int(parallel) > 0 else 8
    slot = derive_slot_id(key, n)
    slots: set[int] = set()
    if slot >= 0:
        slots.add(slot)
    # Also pin slot 0 when parallel==1 path may rewrite — cover common defaults.
    for p in (1, 2, 4, 8, 16, 32):
        s = derive_slot_id(key, p)
        if s >= 0:
            slots.add(s)
    expires_ms = _parse_expires_at(expires_at)
    with _lock:
        _expire_locked(time.time() * 1000)
        _pins[pid] = _Pin(
            pin_id=pid,
            prompt_cache_key=key,
            expires_at_ms=expires_ms,
            slot_ids=slots,
        )
    return {"ok": True, "pin_id": pid, "slot_ids": sorted(slots)}


def unregister_cache_pin(*, pin_id: str = "", prompt_cache_key: str = "") -> dict[str, Any]:
    pid = (pin_id or "").strip()
    key = (prompt_cache_key or "").strip()
    with _lock:
        _expire_locked(time.time() * 1000)
        if pid and pid in _pins:
            del _pins[pid]
            return {"ok": True, "deleted": True}
        if key:
            for p, lease in list(_pins.items()):
                if lease.prompt_cache_key == key:
                    del _pins[p]
                    return {"ok": True, "deleted": True}
    return {"ok": True, "deleted": False}


def pinned_slot_ids(*, now_ms: float | None = None) -> set[int]:
    now = time.time() * 1000 if now_ms is None else now_ms
    with _lock:
        _expire_locked(now)
        out: set[int] = set()
        for p in _pins.values():
            out.update(p.slot_ids)
        return out


def slot_id_from_filename(name: str) -> int | None:
    m = _SLOT_BIN_RE.match(name)
    if not m:
        return None
    return int(m.group(1))


def is_slot_file_pinned(file_name: str, *, now_ms: float | None = None) -> bool:
    sid = slot_id_from_filename(file_name)
    if sid is None:
        return False
    return sid in pinned_slot_ids(now_ms=now_ms)


def pin_ttl_ms_for_file(file_name: str, default_ms: int) -> int:
    """Return extended TTL when the slot is pinned; else default_ms."""
    if is_slot_file_pinned(file_name):
        return max(default_ms, PINNED_SLOT_TTL_MS)
    return default_ms
