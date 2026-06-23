"""Per-request prefill cancellation (Phase 15 v31 engine wiring)."""

from __future__ import annotations

import threading

from runtime.kv.native_decode_loop import prefill_abort_clear, prefill_abort_set


class PrefillCancelToken:
    """Signals chunked prefill to stop after the current page-aligned chunk.

    ``cancel()`` is safe from any thread while ``run_prefill`` runs in C.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        if self._event.is_set():
            return
        self._event.set()
        prefill_abort_set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def reset(self) -> None:
        self._event.clear()
        prefill_abort_clear()
