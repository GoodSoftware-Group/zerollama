"""PagedAttention KV block allocator — Python or native backend (Phase 15)."""

from __future__ import annotations

from typing import Any

from runtime.kv._py_block_pool import SequenceBlockTable
from runtime.kv.backend import (
    block_pool_class,
    block_pool_error_type,
    create_block_pool,
    kv_backend_health,
    kv_backend_name,
)

BlockPoolError = block_pool_error_type()

__all__ = [
    "BlockPool",
    "BlockPoolError",
    "SequenceBlockTable",
    "create_block_pool",
    "kv_backend_name",
    "kv_backend_health",
    "block_pool_class",
]


def __getattr__(name: str) -> Any:
    # Lazy: respect ZEROLLAMA_RUNTIME_KV_NATIVE at access time, not only at first import.
    if name == "BlockPool":
        return block_pool_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
