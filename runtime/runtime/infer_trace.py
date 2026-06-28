"""Opt-in inference phase tracing (Phase 15 debug).

Enable with ``ZEROLLAMA_INFER_TRACE=1``. Logs go to the ``zerollama-runtime``
logger at INFO so they appear in ``/tmp/macos-runtime.log`` during smokes.

WHY separate module: keeps hot paths readable; operators flip one env var
without touching decode logic.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

_log = logging.getLogger("zerollama-runtime")


def infer_trace_enabled() -> bool:
    from runtime.env import infer_trace_enabled as _enabled

    return _enabled()


def infer_trace(event: str, **fields: Any) -> None:
    if not infer_trace_enabled():
        return
    parts = [f"infer_trace {event}"]
    tid = threading.get_ident()
    parts.append(f"tid={tid}")
    for key in sorted(fields):
        val = fields[key]
        if val is None:
            continue
        parts.append(f"{key}={val!r}")
    _log.info(" ".join(parts))


def infer_trace_span(event: str, **fields: Any):
    """Context manager: logs ``enter`` / ``exit`` (+ elapsed ms) for a block."""

    class _Span:
        def __enter__(self):
            self._t0 = time.monotonic()
            infer_trace(f"{event}.enter", **fields)
            return self

        def __exit__(self, exc_type, exc, _tb):
            ms = int((time.monotonic() - self._t0) * 1000)
            if exc_type is None:
                infer_trace(f"{event}.exit", elapsed_ms=ms, **fields)
            else:
                infer_trace(
                    f"{event}.error",
                    elapsed_ms=ms,
                    exc_type=getattr(exc_type, "__name__", str(exc_type)),
                    exc=str(exc) if exc else "",
                    **fields,
                )
            return False

    return _Span()
