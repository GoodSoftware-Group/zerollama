"""Request queues for continuous batching (Phase 2+)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from runtime.gpu.priority import InferencePriority
from runtime.kv.block_pool import BlockPool
from runtime.kv.multi_pool import MultiDeviceBlockTable


class RequestState(str, Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"


@dataclass
class Request:
    request_id: str
    prompt_tokens: list[int]
    max_tokens: int
    state: RequestState = RequestState.WAITING
    generated: list[int] = field(default_factory=list)
    block_table: MultiDeviceBlockTable | None = None
    priority: InferencePriority = InferencePriority.NORMAL
    fifo_seq: int = 0
    gguf: Path | None = None
    num_ctx: int | None = None
    vram_options: dict[str, Any] | None = None
    vram_num_ctx_meta: dict[str, Any] | None = None
    kv_slot: int | None = None
    # L3: stable session key from request options (eliza conversationId, etc.).
    prompt_cache_key: str | None = None
    # vLLM cache_salt — tenant isolation for slot hash + in-process owner key.
    cache_salt: str | None = None
    # When True, kv_slot was derived from prompt_cache_key; llama-server keeps KV
    # after complete() — allocator releases tracking only, not llama slot contents.
    slot_pinned: bool = False
    # HiCache-shaped tier breakdown (SGLang sglext.cached_tokens_details).
    # host = in-process disk slot restore; storage = L3 federated blob restore.
    cached_tokens_host: int = 0
    cached_tokens_storage: int = 0
    cached_tokens_storage_backend: str = ""

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_tokens)

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_tokens) + len(self.generated)


@dataclass
class Scheduler:
    """Minimal PA-aware scheduler; batching loop arrives in Phase 2."""

    kv_pools: list[BlockPool]
    waiting: deque[Request] = field(default_factory=deque)
    running: list[Request] = field(default_factory=list)

    @classmethod
    def for_pools(cls, pools: list[BlockPool]) -> Scheduler:
        if not pools:
            raise ValueError("at least one block pool required")
        return cls(kv_pools=pools)

    def add_request(
        self, req: Request, *, priority: InferencePriority | None = None
    ) -> None:
        if priority is not None:
            req.priority = priority
        if req.fifo_seq == 0:
            from runtime.cross_queue_seq import alloc_cross_queue_seq

            req.fifo_seq = alloc_cross_queue_seq()
        req.block_table = MultiDeviceBlockTable(
            request_id=req.request_id, pools=self.kv_pools
        )
        if req.priority == InferencePriority.HIGH:
            self.waiting.appendleft(req)
        else:
            self.waiting.append(req)

    def pop_waiting(self) -> Request | None:
        if not self.waiting:
            return None
        return self.waiting.popleft()

    def oldest_fifo_seq(self) -> int:
        """Smallest ticket among waiting and in-flight runtime work (for cross-queue FIFO)."""
        seqs = [
            r.fifo_seq
            for r in (*self.waiting, *self.running)
            if r.fifo_seq > 0
        ]
        return min(seqs) if seqs else 0

    def oldest_waiting_fifo_seq(self) -> int:
        """Alias for health/admission; includes running requests."""
        return self.oldest_fifo_seq()

    def pop_waiting_for_tick(
        self, *, low_backpressure: bool, cross_fifo_blocked: bool = False
    ) -> Request | None:
        """Dequeue head unless batch (low) is blocked by inference-first or cross-queue FIFO.

        Why LOW-only stall: enqueue rejects LOW under the same mirrors; NORMAL chat must
        keep running on a shared GPU (see docs/phase11-runtime-admission.md).
        """
        if not self.waiting:
            return None
        head = self.waiting[0]
        if head.priority == InferencePriority.HIGH:
            return self.waiting.popleft()
        if head.priority == InferencePriority.LOW and (
            low_backpressure or cross_fifo_blocked
        ):
            return None
        return self.waiting.popleft()

    def cancel_waiting(self, req: Request) -> bool:
        """Drop a queued request and release its KV blocks. Returns True if it was waiting."""
        try:
            self.waiting.remove(req)
        except ValueError:
            return False
        req.state = RequestState.FINISHED
        if req.block_table is not None:
            req.block_table.release()
            req.block_table = None
        return True

    def finish(self, req: Request) -> None:
        req.state = RequestState.FINISHED
        if req.block_table is not None:
            req.block_table.release()
        if req in self.running:
            self.running.remove(req)
