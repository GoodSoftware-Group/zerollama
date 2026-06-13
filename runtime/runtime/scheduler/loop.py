"""Scheduling tick for continuous batching (Phase 2).

Phase 11: each tick re-runs VRAM/policy checks before KV allocate; pop_waiting_for_tick
stalls only LOW at head under inference-first (normal chat keeps dequeuing).

Phase 15: reserve blocks for ``num_ctx`` when set; assign llama-server ``kv_slot``.
L3: ``try_acquire`` for session-pinned slots (prompt cache key → stable id_slot).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from runtime.gpu.admission import AdmissionMisconfigured, AdmissionRejected
from runtime.gpu.inference_policy import inference_backpressure_blocks_low
from runtime.gpu.priority import InferencePriority
from runtime.gpu.mutex import InferenceGpuCoordinator
from runtime.kv.block_pool import BlockPool, BlockPoolError
from runtime.kv.slots import SlotAllocator
from runtime.scheduler.scheduler import Request, RequestState, Scheduler
from runtime.worker.llama_server import LlamaServerError


@dataclass
class SchedulerLoop:
    """Pulls waiting requests, reserves KV blocks, marks them running."""

    scheduler: Scheduler
    coordinator: InferenceGpuCoordinator
    pools: list[BlockPool] = field(default_factory=list)
    parallel_slots: int = 1
    assign_llama_slots: bool = False
    last_scheduler_tick: int | None = field(default=None, init=False)
    _slots: SlotAllocator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n = max(1, self.parallel_slots)
        self._slots = SlotAllocator(num_slots=n)

    def slot_snapshot(self) -> dict[str, int | list[int]]:
        return self._slots.snapshot()

    def _tokens_to_reserve(self, req: Request) -> int:
        need = req.num_prompt_tokens + req.max_tokens
        if req.num_ctx is not None and req.num_ctx > need:
            need = req.num_ctx
        return need

    def tick(
        self,
        max_admit: int = 4,
        *,
        vram_check: Callable[[Request], None] | None = None,
    ) -> list[Request]:
        """Admit up to ``max_admit`` requests if GPU policy allows."""
        if not self.coordinator.accepts_new_loads():
            return []
        admitted: list[Request] = []
        while len(admitted) < max_admit:
            oldest_fifo = self.scheduler.oldest_fifo_seq()
            low_pressure = inference_backpressure_blocks_low(
                runtime_waiting=len(self.scheduler.waiting),
                runtime_running=len(self.scheduler.running) + len(admitted),
                runtime_oldest_fifo=oldest_fifo,
            )
            from runtime.gpu.inference_policy import cross_fifo_blocks_low

            fifo_blocked = cross_fifo_blocks_low(runtime_oldest_fifo=oldest_fifo)
            req = self.scheduler.pop_waiting_for_tick(
                low_backpressure=low_pressure,
                cross_fifo_blocked=fifo_blocked,
            )
            if req is None:
                if self.scheduler.waiting:
                    head = self.scheduler.waiting[0]
                    if head.priority == InferencePriority.LOW and (
                        low_pressure or fifo_blocked
                    ):
                        break
                break
            if req.block_table is None:
                continue
            if vram_check is not None:
                try:
                    vram_check(req)
                except AdmissionMisconfigured as e:
                    if req.block_table is not None:
                        req.block_table.release()
                        req.block_table = None
                    raise LlamaServerError(str(e)) from e
                except AdmissionRejected:
                    self.scheduler.waiting.appendleft(req)
                    break
                except LlamaServerError:
                    if req.block_table is not None:
                        req.block_table.release()
                        req.block_table = None
                    for prev in admitted:
                        self.complete(prev)
                    raise
            try:
                req.block_table.ensure_capacity(self._tokens_to_reserve(req))
            except BlockPoolError:
                self.scheduler.waiting.appendleft(req)
                break
            except Exception:
                self.scheduler.waiting.appendleft(req)
                break
            if self.assign_llama_slots:
                # L3: pre-assigned kv_slot (session pin) vs dynamic acquire().
                if req.kv_slot is not None and req.kv_slot >= 0:
                    if not self._slots.try_acquire(req.kv_slot):
                        if req.block_table is not None:
                            req.block_table.release()
                            req.block_table = None
                        self.scheduler.waiting.appendleft(req)
                        break
                else:
                    slot = self._slots.acquire()
                    if slot is None:
                        if req.block_table is not None:
                            req.block_table.release()
                            req.block_table = None
                        self.scheduler.waiting.appendleft(req)
                        break
                    req.kv_slot = slot
            req.state = RequestState.PREFILL
            self.scheduler.running.append(req)
            admitted.append(req)
        if admitted:
            from runtime.kv.native_tick import record_scheduler_tick

            self.last_scheduler_tick = record_scheduler_tick()
        return admitted

    def complete(self, req: Request) -> None:
        if req.kv_slot is not None:
            # Pinned sessions: release allocator tracking only. llama-server keeps KV
            # in that id_slot; the next turn re-derives the same slot from cache key.
            self._slots.release(req.kv_slot)
            if not req.slot_pinned:
                req.kv_slot = None
        self.scheduler.finish(req)
