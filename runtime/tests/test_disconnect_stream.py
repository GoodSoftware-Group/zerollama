"""Tests for HTTP disconnect → prefill cancel bridge."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from runtime.kv.native_decode_loop import PrefillAbortedError
from runtime.kv.prefill_cancel import PrefillCancelToken
from runtime.server.disconnect_stream import (
    ndjson_stream_on_disconnect,
    run_sync_on_disconnect,
)


def test_ndjson_stream_on_disconnect_cancels_on_client_close():
    request = MagicMock()
    request.is_disconnected = AsyncMock(side_effect=[False, True, True])

    seen_cancel: list[PrefillCancelToken] = []

    def slow_iter(cancel: PrefillCancelToken):
        seen_cancel.append(cancel)
        yield {"response": "partial", "done": False}
        while not cancel.is_cancelled():
            pass
        yield {"done": True, "done_reason": "cancelled"}

    async def collect() -> list[str]:
        lines: list[str] = []
        async for line in ndjson_stream_on_disconnect(request, slow_iter):
            lines.append(line)
        return lines

    lines = asyncio.run(collect())

    assert seen_cancel
    assert seen_cancel[0].is_cancelled()
    assert any("partial" in ln for ln in lines)


def test_run_sync_on_disconnect_cancels_blocking_call():
    request = MagicMock()
    request.is_disconnected = AsyncMock(side_effect=[False, True, True])

    seen_cancel: list[PrefillCancelToken] = []
    saw_cancelled = False

    def blocking(cancel: PrefillCancelToken) -> str:
        seen_cancel.append(cancel)
        while not cancel.is_cancelled():
            pass
        nonlocal saw_cancelled
        saw_cancelled = True
        raise PrefillAbortedError("KV prefill aborted (client disconnect)")

    async def run() -> None:
        try:
            await run_sync_on_disconnect(request, blocking)
        except PrefillAbortedError:
            return
        raise AssertionError("expected PrefillAbortedError")

    asyncio.run(run())
    assert seen_cancel
    assert saw_cancelled
