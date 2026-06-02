"""Llama sequence KV position vs PA reserve (Phase 15 v4).

Public libllama exposes per-sequence cell positions, not vLLM page handles.
We treat ``pos_max + 1`` (minus SWA ``pos_min``) as physical cells used and verify
they fit the PA page table reserved at admission.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from runtime.kv.bind import primary_block_ids, reserved_token_capacity
from runtime.worker.llama_server import LlamaServerError

if TYPE_CHECKING:
    from runtime.scheduler.scheduler import Request

_log = logging.getLogger(__name__)

_RECENT_ALIGNMENTS: deque[dict[str, Any]] = deque(maxlen=8)


def record_recent_alignment(row: dict[str, Any], *, at: str) -> None:
    """Ring buffer of post-decode PA↔llama rows for ``/health`` debugging."""
    entry = dict(row)
    entry["at"] = at
    entry["ts"] = round(time.time(), 3)
    _RECENT_ALIGNMENTS.append(entry)


def recent_alignments() -> list[dict[str, Any]]:
    return list(_RECENT_ALIGNMENTS)


def clear_recent_alignments_for_tests() -> None:
    _RECENT_ALIGNMENTS.clear()


def physical_strict_enabled() -> bool:
    return os.environ.get("ZEROLLAMA_RUNTIME_KV_PHYSICAL_STRICT", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


@dataclass(frozen=True)
class SequenceKvUsage:
    seq_id: int
    pos_min: int
    pos_max: int

    @property
    def token_cells(self) -> int:
        if self.pos_max < 0:
            return 0
        lo = self.pos_min if self.pos_min >= 0 else 0
        return self.pos_max - lo + 1


def usage_from_libllama(
    lib: Any, ctx: Any, seq_id: int
) -> SequenceKvUsage | None:
    from runtime.worker.libllama_ctypes import sequence_kv_usage as _read

    raw = _read(lib, ctx, seq_id)
    if raw is None:
        return None
    sid, pmin, pmax = raw
    return SequenceKvUsage(seq_id=sid, pos_min=pmin, pos_max=pmax)


def pa_llama_alignment(
    req: Request,
    usage: SequenceKvUsage | None,
    *,
    block_size: int,
) -> dict[str, Any]:
    pa_cap = reserved_token_capacity(req)
    bids = primary_block_ids(req)
    row: dict[str, Any] = {
        "request_id": req.request_id,
        "pa_tokens_reserved": pa_cap,
        "pa_pages_reserved": len(bids),
        "block_size": block_size,
    }
    if usage is None:
        row["llama_tracked"] = False
        return row
    cells = usage.token_cells
    pages_need = (cells + block_size - 1) // block_size if cells > 0 else 0
    row.update(
        {
            "llama_tracked": True,
            "kv_slot": req.kv_slot,
            "seq_id": usage.seq_id,
            "llama_pos_min": usage.pos_min,
            "llama_pos_max": usage.pos_max,
            "llama_token_cells": cells,
            "llama_pages_equivalent": pages_need,
            "aligned": cells <= pa_cap,
            "pages_fit": pages_need <= len(bids) if bids else cells == 0,
        }
    )
    return row


def verify_after_decode(
    req: Request | None,
    usage: SequenceKvUsage | None,
    *,
    block_size: int,
    at: str,
) -> None:
    if req is None or usage is None:
        return
    align = pa_llama_alignment(req, usage, block_size=block_size)
    if not (align.get("aligned") and align.get("pages_fit", True)):
        record_recent_alignment(align, at=at)
    if align.get("aligned") and align.get("pages_fit", True):
        return
    rid = getattr(req, "request_id", None)
    slot = req.kv_slot
    msg = (
        f"KV physical ({at}): llama used {align.get('llama_token_cells')} cells "
        f"but PA reserved {align.get('pa_tokens_reserved')} tokens "
        f"({align.get('pa_pages_reserved')} pages)"
        f"{f', request_id={rid}' if rid else ''}"
        f"{f', kv_slot={slot}' if slot is not None else ''}"
    )
    if physical_strict_enabled():
        raise LlamaServerError(msg)
    _log.warning("%s", msg)


_KV_PHYSICAL_SINGLE_SEQ_NOTE = (
    "live llama seq positions need llama_parallel_slots>1 (shared ctx); "
    "post-decode PA alignment runs on every in-process completion"
)


def kv_physical_health(
    *,
    inprocess_ctx: Any,
    lib: Any,
    running: list[Request],
    block_size: int,
) -> dict[str, Any]:
    rows = [
        {
            "request_id": req.request_id,
            "live_seq_positions": True,
            **pa_llama_alignment(
                req,
                usage_from_libllama(
                    lib, inprocess_ctx, req.kv_slot if req.kv_slot is not None else 0
                ),
                block_size=block_size,
            ),
        }
        for req in running
    ]
    return {
        "bind_level": "seq_position",
        "tensor_pages_bound": False,
        "strict_env": physical_strict_enabled(),
        "running": rows,
    }


def kv_physical_health_pa_only(
    running: list[Request],
    *,
    block_size: int,
) -> dict[str, Any]:
    """In-process loaded with per-request ctx (``llama_parallel_slots==1``)."""
    rows = [
        {
            "request_id": req.request_id,
            "live_seq_positions": False,
            **pa_llama_alignment(req, None, block_size=block_size),
        }
        for req in running
    ]
    return {
        "bind_level": "seq_position",
        "tensor_pages_bound": False,
        "strict_env": physical_strict_enabled(),
        "note": _KV_PHYSICAL_SINGLE_SEQ_NOTE,
        "running": rows,
    }


def kv_bind_physical_level(llama_backend: str, *, inprocess_weights_loaded: bool) -> str | None:
    if llama_backend == "inprocess" and inprocess_weights_loaded:
        return "seq_position"
    return None
