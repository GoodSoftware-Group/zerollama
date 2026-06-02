"""Per-request KV forward plan export (Phase 15 v7).

Serializes the logical page table for operators and future native decode wiring.
Does not bind pages to llama tensors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from runtime.kv.bind import (
    primary_block_ids,
    reserved_token_capacity,
    tokens_to_reserve,
)

if TYPE_CHECKING:
    from runtime.scheduler.scheduler import Request


def kv_forward_plan(req: Request, *, block_size: int) -> dict[str, Any]:
    """Logical page table + slot for one admitted request."""
    bids = primary_block_ids(req)
    pages: list[dict[str, int]] = []
    for i, bid in enumerate(bids):
        start = i * block_size
        pages.append(
            {
                "page": i,
                "block_id": bid,
                "token_start": start,
                "token_end": start + block_size - 1,
            }
        )
    return {
        "request_id": req.request_id,
        "state": req.state.value,
        "kv_slot": req.kv_slot,
        "block_size": block_size,
        "num_ctx": req.num_ctx,
        "pages": pages,
        "pa_tokens_reserved": reserved_token_capacity(req),
        "tokens_to_reserve": tokens_to_reserve(req),
    }


def kv_forward_plans_for_requests(
    requests: list[Request],
    *,
    block_size: int,
) -> list[dict[str, Any]]:
    return [kv_forward_plan(r, block_size=block_size) for r in requests]
