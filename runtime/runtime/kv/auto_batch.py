"""Phase 15 v32 — opt-in auto-batching for concurrent /api/generate (in-process).

WHY: ``generate_batch`` already merges N prompts into one C ``run_batch_step`` path,
but each HTTP ``generate()`` still calls ``completion()`` alone. When several requests
arrive within a short window, coalescing them avoids N separate ``llama_decode`` calls
per token step on the shared multiseq ctx.

Opt-in via ``ZEROLLAMA_KV_AUTO_BATCH=1`` (default off — adds up to
``ZEROLLAMA_KV_AUTO_BATCH_MS`` TTFT latency to wait for co-batch peers).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from runtime.engine import GenerateResult, InferenceEngine
    from runtime.scheduler.scheduler import Request


def auto_batch_window_ms() -> int:
    from runtime.env import kv_auto_batch_window_ms

    return kv_auto_batch_window_ms()


def native_auto_batch_enabled() -> bool:
    """True when env opt-in + native C batch decode is available."""
    from runtime.env import kv_auto_batch_enabled

    if not kv_auto_batch_enabled():
        return False
    from runtime.kv.native_decode_loop import native_batch_decode_available

    return native_batch_decode_available()


def auto_batch_eligible(
    engine: InferenceEngine,
    *,
    gguf: Path | None,
    stream: bool,
) -> bool:
    """Whether this request may enter the auto-batch coordinator."""
    if stream or not native_auto_batch_enabled():
        return False
    from runtime.worker.factory import LlamaBackendKind

    if engine._resolved_llama_backend() != LlamaBackendKind.INPROCESS:
        return False
    if engine._effective_llama_parallel_slots() <= 1:
        return False
    return True


def _options_key(options: dict | None) -> int:
    """Stable hash of sampler options so requests with different settings don't silently batch."""
    if not options:
        return 0
    try:
        return hash(tuple(sorted((k, str(v)) for k, v in options.items())))
    except Exception:
        return 0


def _batch_key(
    gguf: Path | None,
    n_predict: int,
    num_ctx: int | None,
    options: dict | None = None,
) -> tuple[str, int, int | None, int]:
    if gguf is not None:
        try:
            path = str(gguf.resolve())
        except OSError:
            path = str(gguf)
    else:
        path = ""
    return (path, int(n_predict), num_ctx, _options_key(options))


@dataclass
class _PendingJob:
    prompt: str
    request: Request
    n_predict: int
    gguf: Path | None
    num_ctx: int | None
    options: dict | None
    batch_key: tuple[str, int, int | None, int]
    event: threading.Event = field(default_factory=threading.Event)
    result: GenerateResult | None = None
    error: BaseException | None = None


class AutoBatchCoordinator:
    """Collects concurrent admitted requests; flushes via ``completions_parallel``."""

    def __init__(self, engine: InferenceEngine) -> None:
        self._engine = engine
        self._lock = threading.Lock()
        self._pending: dict[tuple[str, int, int | None, int], list[_PendingJob]] = {}
        self._timers: dict[tuple[str, int, int | None, int], threading.Timer] = {}
        self._last_flush_at: float | None = None
        self._flush_count = 0
        self._batched_requests = 0

    def pending_count(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._pending.values())

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": native_auto_batch_enabled(),
            "pending": self.pending_count(),
            "window_ms": auto_batch_window_ms(),
            "flush_count": self._flush_count,
            "batched_requests": self._batched_requests,
            "last_flush_at": self._last_flush_at,
        }

    def submit(
        self,
        *,
        prompt: str,
        request: Request,
        n_predict: int,
        gguf: Path | None,
        num_ctx: int | None,
        options: dict | None,
    ) -> GenerateResult:
        key = _batch_key(gguf, n_predict, num_ctx, options)
        job = _PendingJob(
            prompt=prompt,
            request=request,
            n_predict=n_predict,
            gguf=gguf,
            num_ctx=num_ctx,
            options=options,
            batch_key=key,
        )
        to_flush: list[_PendingJob] | None = None
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
        job.event.wait()
        if job.error is not None:
            raise job.error
        assert job.result is not None
        return job.result

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

    def _flush(self, jobs: list[_PendingJob]) -> None:
        try:
            if len(jobs) == 1:
                results = [
                    self._engine._generate_one_admitted(
                        jobs[0].prompt,
                        jobs[0].request,
                        n_predict=jobs[0].n_predict,
                        gguf=jobs[0].gguf,
                        options=jobs[0].options,
                    )
                ]
            else:
                results = self._engine._generate_parallel_admitted(jobs)
            for job, result in zip(jobs, results):
                job.result = result
                job.event.set()
            with self._lock:
                self._flush_count += 1
                self._batched_requests += len(jobs)
                self._last_flush_at = time.time()
        except BaseException as e:
            for job in jobs:
                job.error = e
                job.event.set()
