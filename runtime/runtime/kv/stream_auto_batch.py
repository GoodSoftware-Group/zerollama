"""Phase 15 v37 — opt-in auto-batching for concurrent streaming ``generate()``.

WHY v32 is not enough: ``AutoBatchCoordinator`` only coalesces non-stream
``generate()`` calls. Concurrent ``/api/generate`` with ``stream=true`` still
decode one row per ``llama_decode`` per token step. v37 applies the same
window/slot-fill policy to streaming requests and demuxes ``seq_idx``-tagged
chunks from ``completions_parallel_stream`` back to each caller's iterator.

Opt-in via ``ZEROLLAMA_KV_AUTO_BATCH_STREAM=1`` (default off — separate from
non-stream ``ZEROLLAMA_KV_AUTO_BATCH`` because streaming batching adds TTFT
latency and interleaved chunk ordering complexity).
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from runtime.kv.auto_batch import _batch_key, auto_batch_window_ms

if TYPE_CHECKING:
    from runtime.engine import InferenceEngine
    from runtime.scheduler.scheduler import Request

_DONE = object()


def native_stream_auto_batch_enabled() -> bool:
    """True when stream opt-in env + native C batch decode is available."""
    from runtime.env import kv_auto_batch_stream_enabled

    if not kv_auto_batch_stream_enabled():
        return False
    from runtime.kv.native_decode_loop import native_batch_decode_available

    return native_batch_decode_available()


def stream_auto_batch_eligible(
    engine: InferenceEngine,
    *,
    gguf: Path | None,
    stream: bool,
) -> bool:
    """Whether this streaming request may enter the stream auto-batch coordinator."""
    if not stream or not native_stream_auto_batch_enabled():
        return False
    from runtime.worker.factory import LlamaBackendKind

    if engine._resolved_llama_backend() != LlamaBackendKind.INPROCESS:
        return False
    if engine._effective_llama_parallel_slots() <= 1:
        return False
    return True


@dataclass
class _PendingStreamJob:
    prompt: str
    request: Request
    n_predict: int
    gguf: Path | None
    num_ctx: int | None
    options: dict | None
    batch_key: tuple[str, int, int | None, int]
    chunk_queue: queue.Queue[Any] = field(default_factory=queue.Queue)
    error: BaseException | None = None


class StreamAutoBatchCoordinator:
    """Collects concurrent streaming requests; flushes via ``completions_parallel_stream``."""

    def __init__(self, engine: InferenceEngine) -> None:
        self._engine = engine
        self._lock = threading.Lock()
        self._pending: dict[tuple[str, int, int | None, int], list[_PendingStreamJob]] = {}
        self._timers: dict[tuple[str, int, int | None, int], threading.Timer] = {}
        self._last_flush_at: float | None = None
        self._flush_count = 0
        self._batched_requests = 0

    def pending_count(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._pending.values())

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": native_stream_auto_batch_enabled(),
            "pending": self.pending_count(),
            "window_ms": auto_batch_window_ms(),
            "flush_count": self._flush_count,
            "batched_requests": self._batched_requests,
            "last_flush_at": self._last_flush_at,
        }

    def iter_stream(
        self,
        *,
        prompt: str,
        request: Request,
        n_predict: int,
        gguf: Path | None,
        num_ctx: int | None,
        options: dict | None,
    ) -> Iterator[dict[str, Any]]:
        """Yield decode-phase chunks for one admitted streaming request."""
        key = _batch_key(gguf, n_predict, num_ctx, options)
        job = _PendingStreamJob(
            prompt=prompt,
            request=request,
            n_predict=n_predict,
            gguf=gguf,
            num_ctx=num_ctx,
            options=options,
            batch_key=key,
        )
        to_flush: list[_PendingStreamJob] | None = None
        max_n = self._engine._effective_llama_parallel_slots()
        with self._lock:
            batch = self._pending.setdefault(key, [])
            batch.append(job)
            if len(batch) >= max_n:
                to_flush = batch[:max_n]
                rest = batch[max_n:]
                if rest:
                    self._pending[key] = rest
                else:
                    self._pending.pop(key, None)
                self._cancel_timer_locked(key)
            elif auto_batch_window_ms() <= 0:
                to_flush = batch[:]
                self._pending.pop(key, None)
                self._cancel_timer_locked(key)
            else:
                self._arm_timer_locked(key)
        if to_flush:
            self._flush(to_flush)
        while True:
            item = job.chunk_queue.get()
            if item is _DONE:
                break
            yield item
        if job.error is not None:
            raise job.error

    def _arm_timer_locked(self, key: tuple[str, int, int | None, int]) -> None:
        if key in self._timers or auto_batch_window_ms() <= 0:
            return
        timer = threading.Timer(
            auto_batch_window_ms() / 1000.0,
            self._timer_flush,
            args=(key,),
        )
        timer.daemon = True
        self._timers[key] = timer
        timer.start()

    def _cancel_timer_locked(self, key: tuple[str, int, int | None, int]) -> None:
        timer = self._timers.pop(key, None)
        if timer is not None:
            timer.cancel()

    def _timer_flush(self, key: tuple[str, int, int | None, int]) -> None:
        with self._lock:
            batch = self._pending.pop(key, [])
            self._timers.pop(key, None)
        if batch:
            self._flush(batch)

    def _flush(self, jobs: list[_PendingStreamJob]) -> None:
        try:
            chunks = self._engine._stream_parallel_admitted(jobs)
            job_by_id = {j.request.request_id: j for j in jobs}
            job_by_idx = {i: j for i, j in enumerate(jobs)}
            for chunk in chunks:
                req_id = chunk.get("request_id")
                job = None
                if req_id is not None:
                    job = job_by_id.get(str(req_id))
                if job is None:
                    seq_idx = int(chunk.get("seq_idx", 0))
                    job = job_by_idx.get(seq_idx)
                if job is not None:
                    job.chunk_queue.put(chunk)
            with self._lock:
                self._flush_count += 1
                self._batched_requests += len(jobs)
                self._last_flush_at = time.time()
        except BaseException as e:
            for job in jobs:
                job.error = e
        finally:
            for job in jobs:
                job.chunk_queue.put(_DONE)
