"""Pure-Python PagedAttention block allocator (fallback when native extension is off)."""

from __future__ import annotations

from dataclasses import dataclass, field


class BlockPoolError(Exception):
    pass


@dataclass
class BlockPool:
    """Fixed-size block pool for KV cache pages.

    Each block holds ``block_size`` logical tokens. Allocation is by block id;
    the engine maps ids to device memory when talking to llama-server.
    """

    num_blocks: int
    block_size: int
    device_id: int = 0
    _free: list[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        self._free = list(range(self.num_blocks - 1, -1, -1))

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def utilization(self) -> float:
        if self.num_blocks == 0:
            return 0.0
        return 1.0 - (self.num_free / self.num_blocks)

    def blocks_for_tokens(self, num_tokens: int) -> int:
        if num_tokens <= 0:
            return 0
        return (num_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, n_blocks: int) -> bool:
        return n_blocks <= self.num_free

    def allocate(self, n_blocks: int) -> list[int]:
        if n_blocks < 0:
            raise ValueError("n_blocks must be non-negative")
        if n_blocks > self.num_free:
            raise BlockPoolError(
                f"device {self.device_id}: need {n_blocks} blocks, {self.num_free} free"
            )
        out: list[int] = []
        for _ in range(n_blocks):
            out.append(self._free.pop())
        return out

    def free(self, block_ids: list[int]) -> None:
        for bid in block_ids:
            if bid < 0 or bid >= self.num_blocks:
                raise ValueError(f"invalid block id {bid}")
            if bid in self._free:
                raise BlockPoolError(f"block {bid} already free")
            self._free.append(bid)

    def reset(self) -> None:
        self._free = list(range(self.num_blocks - 1, -1, -1))


@dataclass
class SequenceBlockTable:
    """Per-request block list (PagedAttention page table)."""

    request_id: str
    pool: BlockPool
    block_ids: list[int] = field(default_factory=list)

    @property
    def num_tokens_capacity(self) -> int:
        return len(self.block_ids) * self.pool.block_size

    def ensure_capacity(self, num_tokens: int) -> None:
        """Grow block list so at least ``num_tokens`` logical slots exist."""
        needed = self.pool.blocks_for_tokens(num_tokens)
        have = len(self.block_ids)
        if needed <= have:
            return
        extra = self.pool.allocate(needed - have)
        self.block_ids.extend(extra)

    def release(self) -> None:
        if self.block_ids:
            self.pool.free(self.block_ids)
            self.block_ids.clear()
