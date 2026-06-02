"""KV pool + scheduler accounting for /health (Phase 15 v1)."""

from __future__ import annotations

from typing import Any

from runtime.kv.bind import primary_block_ids
from runtime.kv.block_pool import BlockPool
from runtime.scheduler.scheduler import Request, Scheduler


def _blocks_for_request(req: Request) -> int:
    if req.block_table is None:
        return 0
    tables = getattr(req.block_table, "_tables", None)
    if not tables:
        return 0
    # TP: same logical blocks on each device — count primary pool only.
    return len(tables[0].block_ids)


def _tokens_capacity_for_request(req: Request, block_size: int) -> int:
    if req.block_table is None:
        return 0
    return req.block_table.num_tokens_capacity


def kv_scheduler_snapshot(
    scheduler: Scheduler,
    pools: list[BlockPool],
    *,
    block_size: int,
    slot_snapshot: dict[str, Any] | None = None,
    llama_parallel_slots: int | None = None,
) -> dict[str, Any]:
    """Summarize reserved KV blocks vs pool capacity."""
    waiting = list(scheduler.waiting)
    running = list(scheduler.running)
    reserved_blocks = sum(_blocks_for_request(r) for r in (*waiting, *running))
    total_blocks = sum(p.num_blocks for p in pools)
    free_blocks = sum(p.num_free for p in pools)

    def _req_row(req: Request) -> dict[str, Any]:
        row: dict[str, Any] = {
            "request_id": req.request_id,
            "state": req.state.value,
            "blocks": _blocks_for_request(req),
            "tokens_capacity": _tokens_capacity_for_request(req, block_size),
            "num_ctx": req.num_ctx,
        }
        if req.kv_slot is not None:
            row["kv_slot"] = req.kv_slot
        bids = primary_block_ids(req)
        if bids:
            row["block_ids"] = bids
        return row

    out: dict[str, Any] = {
        "block_size": block_size,
        "blocks_total": total_blocks,
        "blocks_free": free_blocks,
        "blocks_reserved": reserved_blocks,
        "utilization": round(
            (reserved_blocks / total_blocks) if total_blocks else 0.0, 4
        ),
        "waiting": len(waiting),
        "running": len(running),
        "requests": [_req_row(r) for r in (*waiting, *running)],
    }
    if slot_snapshot is not None:
        out["llama_slots"] = slot_snapshot
    if llama_parallel_slots is not None:
        out["llama_parallel_slots"] = llama_parallel_slots
    return out
