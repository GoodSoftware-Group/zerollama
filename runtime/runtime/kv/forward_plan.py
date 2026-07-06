"""Per-request KV forward plan export (Phase 15 v7).

Serializes the logical page table for operators and future native decode wiring.
v9 adds ``decode_prefill`` (planned page-aligned prefill batches) when admitted.
v10 adds in-progress plans when ``current_pos`` is known (live llama seq position).
v28 adds ``kv_continuous_batch_forward_plan`` — merged decode-step preview for
N running sequences (``/health.kv_continuous_batch``).
v20a adds ``native_page_table`` + ``page_table_native_parity`` from the C registry
when admitted (operator debug for v19 tensor bind scaffold). Does not bind pages
to llama tensors.
v40 adds ``page_migration_summary`` on running admitted plans when tensor/physical
bind probe data is available (lightweight; full plan on ``/internal/kv-snapshot``).

WHY ``decode_prefill`` is admit-only
------------------------------------
Waiting requests have no ``block_ids`` yet — exporting a prefill plan without a
reserved page table would show chunk boundaries that are not yet backed by PA
blocks.  Operators use ``pages[]`` + ``decode_prefill`` together.

WHY ``current_pos`` is optional (v10)
-------------------------------------
Live seq positions need a shared in-process ``llama_context`` (multi-seq) or an
active decode.  Without ``kv_physical`` data we keep the v9 admit-time plan
(``pos_start=0``).  Single-seq per-request ctx does not expose live positions
on ``/health`` unless ``ZEROLLAMA_RUNTIME_KV_LIVE_PHYSICAL=1``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from runtime.kv.bind import (
    primary_block_ids,
    reserved_token_capacity,
    tokens_to_reserve,
)
from runtime.kv.decode_plan import (
    kv_continuous_batch_step_plan,
    kv_decode_prefill_plan,
    kv_decode_step_plan,
    kv_decode_work_plan,
)

if TYPE_CHECKING:
    from runtime.scheduler.scheduler import Request


def kv_forward_plan(
    req: Request,
    *,
    block_size: int,
    current_pos: int | None = None,
) -> dict[str, Any]:
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
    plan: dict[str, Any] = {
        "request_id": req.request_id,
        "state": req.state.value,
        "kv_slot": req.kv_slot,
        "block_size": block_size,
        "num_ctx": req.num_ctx,
        "pages": pages,
        "pa_tokens_reserved": reserved_token_capacity(req),
        "tokens_to_reserve": tokens_to_reserve(req),
    }
    # v9–v10: planned prefill + decode batches (future native decode loop).
    # WHY prompt_tokens and bids: need both a prompt and an admitted block table —
    # waiting requests have tokens but no block_ids yet.
    # WHY seq_id == kv_slot: SlotAllocator assigns slot N → in-process seq N; subprocess
    # maps the same integer to id_slot.  If these ever diverge, pass the real seq_id here.
    if req.prompt_tokens and bids:
        pos_start = int(current_pos) if current_pos is not None else 0
        plan["decode_prefill"] = kv_decode_prefill_plan(
            list(req.prompt_tokens),
            block_size=block_size,
            kv_slot=req.kv_slot,
            seq_id=int(req.kv_slot or 0),
            pos_start=pos_start,
        )
        plan["decode_work"] = kv_decode_work_plan(
            list(req.prompt_tokens),
            block_size=block_size,
            max_tokens=req.max_tokens,
            kv_slot=req.kv_slot,
            seq_id=int(req.kv_slot or 0),
            current_pos=current_pos,
        )
        # WHY decode_steps removed: decode_work.decode carries the same object.
        # plan_current_pos is kept as a top-level scalar for quick access without
        # unpacking decode_work.
        if current_pos is not None:
            plan["plan_current_pos"] = pos_start
        # v20a: mirror native C page_bind registry for operator / snapshot parity.
        if req.kv_slot is not None and req.kv_slot >= 0:
            from runtime.kv.tensor_probe import (
                export_page_table,
                page_table_native_parity,
            )

            native_rows = export_page_table(int(req.kv_slot))
            if native_rows:
                plan["native_page_table"] = native_rows
                plan["page_table_native_parity"] = page_table_native_parity(
                    pages, native_rows
                )
    return plan


def kv_forward_plans_for_requests(
    requests: list[Request],
    *,
    block_size: int,
    current_pos_by_request_id: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    pos_map = current_pos_by_request_id or {}
    return [
        kv_forward_plan(
            r,
            block_size=block_size,
            current_pos=pos_map.get(r.request_id),
        )
        for r in requests
    ]


def kv_continuous_batch_forward_plan(
    requests: list[Request],
    *,
    block_size: int,
    current_pos_by_request_id: dict[str, int] | None = None,
    parallel_slots: int = 1,
) -> dict[str, Any]:
    """Export the next continuous-batch decode step for running requests (v28).

    WHY separate from per-request ``kv_forward_plan``: v27 merges N decode rows
    into one ``run_batch_step``; operators need a single merged view on ``/health``
    to verify batching would fire before GPU sign-off.

    ``token`` is ``0`` with ``token_placeholder=True`` — the in-flight feed token
    is not tracked on ``Request`` during decode; seq_id/pos/kv_slot are the fields
    that matter for batch layout and page-bind validation.
    """
    from runtime.kv.native_decode_loop import native_batch_decode_available

    pos_map = current_pos_by_request_id or {}
    entries: list[dict[str, Any]] = []
    for req in requests:
        slot = req.kv_slot
        if slot is None or slot < 0:
            continue
        pos = pos_map.get(req.request_id)
        if pos is None:
            continue
        n_prompt = len(req.prompt_tokens)
        if pos < n_prompt:
            continue
        tokens_generated = pos - n_prompt
        if tokens_generated >= req.max_tokens:
            continue
        entries.append(
            {
                "token": 0,
                "token_placeholder": True,
                "seq_id": int(slot),
                "pos": int(pos),
                "kv_slot": int(slot),
                "request_id": req.request_id,
                "tokens_generated": tokens_generated,
                "decode_remaining": req.max_tokens - tokens_generated,
            }
        )

    step_plan = (
        kv_continuous_batch_step_plan(entries, block_size=block_size)
        if entries
        else {
            "n_batch_rows": 0,
            "layout_source": "python",
            "rows": [],
        }
    )
    batch_available = native_batch_decode_available()
    n_rows = len(entries)
    would_batch = batch_available and parallel_slots > 1 and n_rows >= 2
    note: str | None = None
    if not batch_available:
        note = "native batch decode not linked or ZEROLLAMA_KV_NATIVE_BATCH=0"
    elif parallel_slots <= 1:
        note = "llama_parallel_slots<=1; batch path requires shared multi-seq ctx"
    elif n_rows < 2:
        note = "fewer than 2 running decode-phase requests"

    return {
        "batch_available": batch_available,
        "parallel_slots": parallel_slots,
        "n_decode_candidates": n_rows,
        "would_batch": would_batch,
        "step_plan": step_plan,
        "note": note,
    }
