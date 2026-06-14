"""Phase 15 v9–v10 — decode plan export from forward plans.

WHY this module
---------------
v8 builds ``llama_batch`` metadata in C and page-chunks long prefills at decode time.
Operators and future native decode loops need the **same plan** visible on
``/health`` / ``/internal/kv-snapshot`` without running inference — bridges
``kv_forward_plans`` (logical pages) to concrete prefill batch boundaries.

WHY shared chunk helpers (v23)
------------------------------
``iter_prefill_execute_chunks`` is the single source for ctypes prefill batch
boundaries **and** ``kv_decode_prefill_plan`` export.  ``libllama_ctypes._prefill_prompt``
calls it directly so ``/health`` ``decode_prefill`` matches runtime decode.

WHY ``pos_start=0`` at admit (v9)
----------------------------------
The default plan covers the *full prompt* as seen at admit time
(``req.prompt_tokens``).  v10 adds optional ``current_pos`` (next llama write
position) for running requests so operators see **remaining** prefill + decode
steps when live seq positions are available from ``kv_physical``.
"""

from __future__ import annotations

from typing import Any

from runtime.kv.bind import logical_page_for_token
from runtime.kv.native_decode_batch import (
    iter_prefill_decode_chunks,
    native_decode_batch_available,
)


def next_pos_from_llama(pos_max: int) -> int:
    """Next token write position from libllama ``llama_memory_seq_pos_max``.

    WHY pos_max + 1: ``pos_max`` is the highest position already written; the next
    ``llama_decode`` batch starts at ``pos_max + 1``.  Returns 0 when no cells
    exist yet (``pos_max < 0``).
    """
    if pos_max < 0:
        return 0
    return pos_max + 1


def iter_prefill_execute_chunks(
    prompt_tokens: list[int],
    *,
    block_size: int,
    pos_start: int = 0,
) -> list[tuple[list[int], int, bool]]:
    """Prefill execution tuples: ``(tokens, chunk_pos_start, logits_last)``.

    WHY final chunk ``logits_last=True``: v15+ requires valid logits on the last
    prefill token before the first ``llama_sampler_sample`` (native or ctypes).
    Intermediate page-aligned chunks use ``logits_last=False``.
    """
    if not prompt_tokens or pos_start >= len(prompt_tokens):
        return []
    remaining = list(prompt_tokens[pos_start:])
    chunk_pos = pos_start
    if block_size <= 0 or len(remaining) <= block_size:
        return [(remaining, chunk_pos, True)]
    if not native_decode_batch_available():
        return [(remaining, chunk_pos, True)]
    chunks = iter_prefill_decode_chunks(
        remaining,
        block_size=block_size,
        pos_start=chunk_pos,
    )
    if len(chunks) <= 1:
        tokens, pos = chunks[0]
        return [(tokens, pos, True)]
    last = len(chunks) - 1
    return [
        (tokens, pos, idx == last)
        for idx, (tokens, pos) in enumerate(chunks)
    ]


def kv_decode_prefill_plan(
    prompt_tokens: list[int],
    *,
    block_size: int,
    kv_slot: int | None = None,
    seq_id: int = 0,
    pos_start: int = 0,
) -> dict[str, Any]:
    """Planned prefill batches for ``prompt_tokens`` (mirrors ctypes execution).

    Uses ``iter_prefill_execute_chunks`` so ``logits_last`` on each exported chunk
    matches ``libllama_ctypes._prefill_prompt``.

    ``pos_start`` (v10): next llama write position.  When ``0``, plan the full
    prompt (admit-time default).  When ``> 0``, plan only remaining prompt tokens
    from that position — matches what ctypes would prefill on resume.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if pos_start < 0:
        raise ValueError("pos_start must be non-negative")
    if not prompt_tokens:
        return _empty_prefill_plan(kv_slot=kv_slot, pos_start=pos_start)

    n_prompt = len(prompt_tokens)
    if pos_start >= n_prompt:
        return {
            "prefill_chunks": [],
            "n_prefill_batches": 0,
            "prefill_complete": True,
            "pos_start": pos_start,
            "layout_source": _layout_source(),
            "page_bind_slot": kv_slot,
        }

    exec_chunks = iter_prefill_execute_chunks(
        prompt_tokens,
        block_size=block_size,
        pos_start=pos_start,
    )
    prefill_chunks: list[dict[str, Any]] = []
    for idx, (tokens, chunk_pos_start, logits_last) in enumerate(exec_chunks):
        pos_end = chunk_pos_start + len(tokens) - 1
        entry: dict[str, Any] = {
            "chunk_index": idx,
            "token_count": len(tokens),
            "pos_start": chunk_pos_start,
            "pos_end": pos_end,
            "logits_last": logits_last,
        }
        page_start, _ = logical_page_for_token(chunk_pos_start, block_size)
        page_end, _ = logical_page_for_token(pos_end, block_size)
        # WHY page_range: ties each chunk to logical pages in kv_forward_plans.pages[]
        entry["page_range"] = [page_start, page_end]
        layout = _batch_layout_summary(
            tokens, seq_id=seq_id, pos_start=chunk_pos_start
        )
        if layout is not None:
            entry["batch_layout"] = layout
        prefill_chunks.append(entry)

    out: dict[str, Any] = {
        "prefill_chunks": prefill_chunks,
        "n_prefill_batches": len(prefill_chunks),
        "layout_source": _layout_source(),
        "page_bind_slot": kv_slot,
    }
    if pos_start > 0:
        out["pos_start"] = pos_start
        out["prefill_complete"] = False
    return out


def kv_decode_step_plan(
    *,
    current_pos: int,
    n_prompt: int,
    max_tokens: int,
    block_size: int,
    kv_slot: int | None = None,
    seq_id: int = 0,
) -> dict[str, Any]:
    """Planned decode batches from ``current_pos`` (export-only).

    WHY single-token steps with ``logits_last=True``: matches
    ``libllama_ctypes._decode_stream`` — each generation step builds a one-token
    batch at ``pos_start=n_pos`` with logits on the last (only) token.

    WHY omitted when ``current_pos < n_prompt``: prefill is not finished; decode
    steps are not meaningful until all prompt tokens are written.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if current_pos < 0:
        raise ValueError("current_pos must be non-negative")
    if max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")

    if current_pos < n_prompt:
        return {
            "pending_prefill": True,
            "current_pos": current_pos,
            "n_prompt": n_prompt,
            "page_bind_slot": kv_slot,
        }

    tokens_generated = current_pos - n_prompt
    remaining = max(0, max_tokens - tokens_generated)
    step: dict[str, Any] = {
        "token_count": 1,
        "logits_last": True,
    }
    layout = _batch_layout_summary([0], seq_id=seq_id, pos_start=current_pos)
    if layout is not None:
        step["batch_layout"] = layout

    pos_end = current_pos + remaining - 1 if remaining > 0 else current_pos - 1
    page_start, _ = logical_page_for_token(current_pos, block_size)
    page_end, _ = logical_page_for_token(max(current_pos, pos_end), block_size)

    return {
        "current_pos": current_pos,
        "n_prompt": n_prompt,
        "tokens_generated": tokens_generated,
        "n_decode_batches_remaining": remaining,
        "pos_range": [current_pos, pos_end] if remaining > 0 else None,
        "page_range": [page_start, page_end] if remaining > 0 else None,
        "step": step,
        "layout_source": _layout_source(),
        "page_bind_slot": kv_slot,
    }


def kv_decode_work_plan(
    prompt_tokens: list[int],
    *,
    block_size: int,
    max_tokens: int,
    kv_slot: int | None = None,
    seq_id: int = 0,
    current_pos: int | None = None,
) -> dict[str, Any]:
    """Unified prefill + decode plan for operators and a future native decode loop.

    WHY one object: v9 ``decode_prefill`` and v10 ``decode_steps`` are separate slices;
    consumers (``/health``, future C loop) need a single phase indicator without
    inferring it from two optional sub-objects.

    ``phase`` values:
    - ``admit`` — no live position; full prompt prefill plan (v9 default).
    - ``prefill`` — running; more prompt tokens remain at ``current_pos``.
    - ``decode`` — prefill complete; generation steps remain.
    - ``done`` — prefill complete and no decode batches left.
    """
    pos_start = int(current_pos) if current_pos is not None else 0
    n_prompt = len(prompt_tokens)
    prefill = kv_decode_prefill_plan(
        prompt_tokens,
        block_size=block_size,
        kv_slot=kv_slot,
        seq_id=seq_id,
        pos_start=pos_start,
    )
    if current_pos is None:
        return {
            "phase": "admit",
            "current_pos": 0,
            "prefill": prefill,
        }

    decode = kv_decode_step_plan(
        current_pos=pos_start,
        n_prompt=n_prompt,
        max_tokens=max_tokens,
        block_size=block_size,
        kv_slot=kv_slot,
        seq_id=seq_id,
    )
    if prefill.get("prefill_complete"):
        phase = "done" if decode.get("n_decode_batches_remaining", 0) == 0 else "decode"
    elif decode.get("pending_prefill"):
        phase = "prefill"
    else:
        phase = "decode"

    return {
        "phase": phase,
        "current_pos": pos_start,
        "prefill": prefill,
        "decode": decode,
    }


def _empty_prefill_plan(*, kv_slot: int | None, pos_start: int) -> dict[str, Any]:
    # WHY prefill_complete always True: no prompt tokens → nothing to prefill regardless
    # of pos_start.  Omitting the flag would leave kv_decode_work_plan phase logic
    # ambiguous (None is falsy, matching the non-complete branch).
    out: dict[str, Any] = {
        "prefill_chunks": [],
        "n_prefill_batches": 0,
        "prefill_complete": True,
        "layout_source": _layout_source(),
        "page_bind_slot": kv_slot,
    }
    if pos_start > 0:
        out["pos_start"] = pos_start
    return out


def _layout_source() -> str:
    return "native" if native_decode_batch_available() else "python"


def _batch_layout_summary(
    tokens: list[int],
    *,
    seq_id: int,
    pos_start: int,
) -> dict[str, Any] | None:
    if not native_decode_batch_available():
        return None
    from runtime.kv._kv_native import decode_batch_layout

    # logits_last=0 for prefill summaries; decode step plan passes a single-token
    # batch and uses logits_last=1 in the real path — summary uses pos only here.
    logits_flag = 1 if len(tokens) == 1 else 0
    layout = decode_batch_layout(
        tokens, int(seq_id), int(pos_start), logits_flag
    )
    pos = layout.get("pos") or []
    if not pos:
        return {"n_tokens": len(tokens), "first_pos": pos_start, "last_pos": pos_start}
    return {
        "n_tokens": len(tokens),
        "first_pos": int(pos[0]),
        "last_pos": int(pos[-1]),
    }


def kv_continuous_batch_step_plan(
    entries: list[dict[str, int]],
    *,
    block_size: int,
) -> dict[str, Any]:
    """Export one continuous-batch decode step for N active sequences (v26).

    Each entry must include ``token``, ``seq_id``, and ``pos`` (llama write
    position for this step).  Optional ``kv_slot`` is echoed for operators.

    WHY separate from ``kv_decode_step_plan``: continuous batching merges N
    single-token rows into one ``llama_decode`` when ``llama_parallel_slots>1``.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not entries:
        return {
            "n_batch_rows": 0,
            "layout_source": _layout_source(),
            "rows": [],
        }

    tokens = [int(e["token"]) for e in entries]
    seq_ids = [int(e["seq_id"]) for e in entries]
    positions = [int(e["pos"]) for e in entries]
    layout = None
    if native_decode_batch_available():
        from runtime.kv._kv_native import decode_batch_layout_multi

        layout = decode_batch_layout_multi(tokens, seq_ids, positions)

    rows: list[dict[str, Any]] = []
    for i, entry in enumerate(entries):
        pos = positions[i]
        page, _ = logical_page_for_token(pos, block_size)
        row: dict[str, Any] = {
            "token": tokens[i],
            "seq_id": seq_ids[i],
            "pos": pos,
            "page": page,
            "kv_slot": entry.get("kv_slot", seq_ids[i]),
        }
        rows.append(row)

    return {
        "n_batch_rows": len(rows),
        "layout_source": _layout_source(),
        "batch_layout": layout,
        "rows": rows,
    }
