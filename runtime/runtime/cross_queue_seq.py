"""Global cross-queue FIFO tickets allocated by the Go daemon."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

from runtime.go_internal_url import connectable_go_base_url

_alloc_lock = threading.Lock()
_local_fallback = 0


def alloc_cross_queue_seq() -> int:
    """Return the next global FIFO ticket from Go, or 0 when unavailable (fail-open)."""
    url = f"{connectable_go_base_url()}/internal/cross-queue-seq"
    req = urllib.request.Request(
        url,
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=0.5) as resp:
            body = json.loads(resp.read().decode())
        seq = int(body.get("seq", 0))
        return seq if seq > 0 else 0
    except (OSError, urllib.error.URLError, ValueError, TypeError, json.JSONDecodeError):
        return _local_fallback_seq()


def _local_fallback_seq() -> int:
    """High-range local tickets when Go is unreachable (keeps ordering among runtime-only work)."""
    global _local_fallback
    with _alloc_lock:
        if _local_fallback < (1 << 48):
            _local_fallback = 1 << 48
        _local_fallback += 1
        return _local_fallback
