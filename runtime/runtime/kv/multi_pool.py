"""KV block tables spanning multiple GPU block pools (tensor-parallel layout)."""

from __future__ import annotations

from dataclasses import dataclass, field

from runtime.kv.block_pool import BlockPool, SequenceBlockTable


@dataclass
class MultiDeviceBlockTable:
    """Reserve matching capacity on each device pool (TP / layer-split)."""

    request_id: str
    pools: list[BlockPool]
    device_ids: list[int] = field(default_factory=list)
    _tables: list[SequenceBlockTable] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not self.pools:
            raise ValueError("pools must be non-empty")
        if not self.device_ids:
            self.device_ids = [p.device_id for p in self.pools]
        self._tables = [
            SequenceBlockTable(request_id=self.request_id, pool=p) for p in self.pools
        ]

    @property
    def num_tokens_capacity(self) -> int:
        if not self._tables:
            return 0
        return min(t.num_tokens_capacity for t in self._tables)

    def ensure_capacity(self, num_tokens: int) -> None:
        for table in self._tables:
            table.ensure_capacity(num_tokens)

    def release(self) -> None:
        for table in self._tables:
            table.release()
