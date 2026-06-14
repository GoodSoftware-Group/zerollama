"""Native decode batch layout (Phase 15 v8 — C-built llama_batch fields).

Generation still calls ``llama_decode`` from Python ctypes; C builds batch metadata
and page-aligned prefill chunks wired to registered page tables.
"""

from __future__ import annotations

import ctypes
from typing import Any

from runtime.worker.libllama_ctypes import LLAMA_TOKEN, LlamaBatch, LlamaServerError


def native_decode_batch_available() -> bool:
    try:
        from runtime.kv._kv_native import decode_batch_layout  # noqa: F401

        return True
    except ImportError:
        return False
    except Exception:
        return False


def build_batch_from_tokens(
    lib: Any,
    tokens: list[int],
    *,
    seq_id: int,
    n_seq_max: int,
    logits_last: bool,
    pos_start: int = 0,
    kv_slot: int | None = None,
) -> LlamaBatch:
    """Build a heap ``LlamaBatch`` using native C layout when available.

    Why C layout when ext built: same fields as ``_batch_from_tokens`` but built
    in the extension to keep pos/seq_id/logits lists off the interpreter hot path.
    Page-bind validation runs here once per batch (not in the chunk iterator).
    """
    if kv_slot is not None:
        from runtime.kv.page_bind import validate_token_positions

        validate_token_positions(kv_slot, pos_start, len(tokens))

    if native_decode_batch_available():
        from runtime.kv._kv_native import decode_batch_layout

        layout = decode_batch_layout(
            tokens, int(seq_id), int(pos_start), 1 if logits_last else 0
        )
        n = len(layout["token"])
        batch = lib.llama_batch_init(
            ctypes.c_int32(n), ctypes.c_int32(0), ctypes.c_int32(n_seq_max)
        )
        batch.n_tokens = n
        for i in range(n):
            batch.token[i] = LLAMA_TOKEN(int(layout["token"][i]))
            batch.pos[i] = int(layout["pos"][i])
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = ctypes.c_int32(int(layout["seq_id"][i]))
            batch.logits[i] = int(layout["logits"][i])
        return batch

    from runtime.worker.libllama_ctypes import _batch_from_tokens

    return _batch_from_tokens(
        lib,
        tokens,
        seq_id=seq_id,
        n_seq_max=n_seq_max,
        logits_last=logits_last,
        pos_start=pos_start,
    )


def iter_prefill_decode_chunks(
    tokens: list[int],
    *,
    block_size: int,
    pos_start: int = 0,
) -> list[tuple[list[int], int]]:
    """Page-aligned prefill chunks as (tokens, chunk_pos_start) pairs.

    Page-bind validation runs in ``build_batch_from_tokens`` when batches are built.

    WHY shared with v9 decode plan export: ``kv_decode_prefill_plan`` calls this
    same function so exported ``decode_prefill`` chunk boundaries match what
    ``libllama_ctypes._prefill_prompt`` executes at decode time.
    """
    if not tokens:
        raise LlamaServerError("empty token batch")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    if native_decode_batch_available():
        from runtime.kv._kv_native import decode_prefill_chunks

        raw = decode_prefill_chunks(tokens, int(block_size), int(pos_start))
        return [([int(t) for t in chunk_tokens], int(chunk_pos)) for chunk_tokens, chunk_pos in raw]

    return [(list(tokens), pos_start)]
