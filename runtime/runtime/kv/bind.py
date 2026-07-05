"""PA block tables ↔ llama sequence/slot binding (Phase 15 v3).

llama.cpp does not expose vLLM-style paged KV block handles on the public API.
This module defines the **logical contract**: reserved ``block_ids`` are page indices
(sequence page *i* → pool block id ``block_ids[i]``, not ``block_ids[i] == i``), and
``kv_slot`` / ``seq_id`` selects the llama
sequence. Physical tensor pages are still owned by llama until a later native bind.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from runtime.worker.llama_server import LlamaServerError

if TYPE_CHECKING:
    from runtime.scheduler.scheduler import Request


class KvBindMode(str, Enum):
    """How scheduler block ids relate to llama KV today."""

    ACCOUNTING_ONLY = "accounting_only"
    """Block pool tracks capacity only; llama manages its own KV layout."""

    SEQ_SLOT = "seq_slot"
    """``kv_slot`` maps to subprocess ``id_slot`` or in-process ``seq_id``."""


def primary_block_ids(req: Request) -> list[int]:
    """Per-sequence page table on the primary KV pool (device 0).

    ``block_ids[i]`` is the pool block id for the *i*-th page (token range
    ``[i * block_size, (i + 1) * block_size)``). Ids are allocator-chosen, not
    necessarily equal to ``i``.
    """
    table = req.block_table
    if table is None:
        return []
    tables = getattr(table, "_tables", None)
    if not tables:
        return []
    return list(tables[0].block_ids)


def reserved_token_capacity(req: Request) -> int:
    if req.block_table is None:
        return 0
    return req.block_table.num_tokens_capacity


def tokens_to_reserve(req: Request) -> int:
    """Same rule as ``SchedulerLoop._tokens_to_reserve``."""
    need = req.num_prompt_tokens + req.max_tokens
    if req.num_ctx is not None and req.num_ctx > need:
        need = req.num_ctx
    return need


def logical_page_for_token(token_pos: int, block_size: int) -> tuple[int, int]:
    """Map a token index to (page_index, offset_within_page)."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if token_pos < 0:
        raise ValueError("token_pos must be non-negative")
    page = token_pos // block_size
    return page, token_pos % block_size


def assert_kv_capacity(req: Request, *, block_size: int, at: str) -> None:
    """Fail fast if admission did not reserve enough PA pages for this request."""
    need = tokens_to_reserve(req)
    have = reserved_token_capacity(req)
    if need > have:
        pages_need = (need + block_size - 1) // block_size
        pages_have = len(primary_block_ids(req))
        raise LlamaServerError(
            f"KV bind ({at}): need {need} tokens ({pages_need} pages) but "
            f"reserved {have} ({pages_have} block_ids)"
        )


def kv_bind_health(
    *,
    llama_backend: str,
    assign_llama_slots: bool,
    parallel_slots: int,
    physical_bind_level: str | None = None,
    physical_pages_bound: bool = False,
) -> dict[str, Any]:
    mode = KvBindMode.ACCOUNTING_ONLY
    if assign_llama_slots and parallel_slots >= 1:
        mode = KvBindMode.SEQ_SLOT
    return {
        "mode": mode.value,
        "physical_pages_bound": physical_pages_bound,
        "physical_bind_level": physical_bind_level,
        "page_table_semantics": (
            "block_ids[i] is pool block id for sequence page i (ordinal page, not id==i)"
        ),
        "llama_backend": llama_backend,
        "parallel_slots": parallel_slots,
    }
