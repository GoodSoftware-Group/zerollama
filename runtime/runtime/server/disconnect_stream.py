"""Bridge sync inference iterators to async HTTP with client-disconnect cancel."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from runtime.kv.native_decode_loop import PrefillAbortedError
from runtime.kv.prefill_cancel import PrefillCancelToken
from runtime.worker.llama_server import LlamaServerError


def _producer(
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue[Any | None],
    sync_iter_fn: Callable[[PrefillCancelToken], Iterator[Any]],
    cancel: PrefillCancelToken,
) -> None:
    try:
        for item in sync_iter_fn(cancel):
            asyncio.run_coroutine_threadsafe(queue.put(item), loop).result()
    except PrefillAbortedError:
        asyncio.run_coroutine_threadsafe(
            queue.put(
                {
                    "error": "request cancelled",
                    "done": True,
                    "done_reason": "cancelled",
                }
            ),
            loop,
        ).result()
    except LlamaServerError as exc:
        asyncio.run_coroutine_threadsafe(
            queue.put({"error": str(exc)}),
            loop,
        ).result()
    except Exception as exc:
        asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
    finally:
        cancel.reset()
        asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()


async def _stream_on_disconnect(
    request: Any,
    sync_iter_fn: Callable[[PrefillCancelToken], Iterator[Any]],
    *,
    encode: Callable[[Any], str] | None = None,
) -> AsyncIterator[str]:
    """Yield encoded chunks from a sync iterator; cancel prefill on disconnect."""
    cancel = PrefillCancelToken()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Any | None] = asyncio.Queue()
    loop.run_in_executor(
        None,
        _producer,
        loop,
        queue,
        sync_iter_fn,
        cancel,
    )
    encode = encode or (lambda item: json.dumps(item) + "\n")

    while True:
        if await request.is_disconnected():
            cancel.cancel()
        try:
            item = await asyncio.wait_for(queue.get(), timeout=0.05)
        except asyncio.TimeoutError:
            continue
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        yield encode(item)


async def ndjson_stream_on_disconnect(
    request: Any,
    sync_iter_fn: Callable[[PrefillCancelToken], Iterator[Any]],
) -> AsyncIterator[str]:
    """Yield NDJSON lines from a sync iterator; cancel prefill on HTTP disconnect."""
    async for line in _stream_on_disconnect(request, sync_iter_fn):
        yield line


async def sse_stream_on_disconnect(
    request: Any,
    sync_iter_fn: Callable[[PrefillCancelToken], Iterator[str]],
) -> AsyncIterator[str]:
    """Yield raw SSE lines; cancel prefill on HTTP disconnect."""
    async for line in _stream_on_disconnect(request, sync_iter_fn, encode=lambda s: s):
        yield line


async def run_sync_on_disconnect(
    request: Any,
    fn: Callable[[PrefillCancelToken], Any],
) -> Any:
    """Run a blocking inference call; cancel native prefill if the client disconnects.

    WHY: streaming endpoints already abort long prefills on disconnect; non-stream
    /api/generate and /api/chat could hold the GPU until prefill finished even when
    the HTTP client had closed (agents timing out on long context).
    """
    cancel = PrefillCancelToken()
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(None, lambda: fn(cancel))
    try:
        while not fut.done():
            if await request.is_disconnected():
                cancel.cancel()
            await asyncio.sleep(0.05)
        return await fut
    finally:
        cancel.reset()
