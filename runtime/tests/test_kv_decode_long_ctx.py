"""Phase 15 v25 — 131k long-context decode plan + page-bind validation.

WHY synthetic (no GPU): validates chunk math and bind boundaries at 131072 ctx
before operators run L2 fork-only 131k bench legs. Catches page-table cap and
per-chunk bind validation regressions without loading a model.
"""

from __future__ import annotations

import pytest

from runtime.kv.backend import native_available
from runtime.kv.decode_plan import iter_prefill_execute_chunks, kv_decode_prefill_plan
from runtime.kv.native_decode_batch import native_decode_batch_available

NUM_CTX = 131072
BLOCK_SIZE = 16
N_PAGES = NUM_CTX // BLOCK_SIZE  # 8192


def test_prefill_plan_131k_chunk_count():
    """131072 tokens at block_size=16 → 8192 page-aligned prefill chunks."""
    if not native_decode_batch_available():
        pytest.skip("native batch ext not built")
    tokens = [0] * NUM_CTX
    plan = kv_decode_prefill_plan(tokens, block_size=BLOCK_SIZE, kv_slot=0)
    assert plan["n_prefill_batches"] == N_PAGES
    chunks = plan["prefill_chunks"]
    assert len(chunks) == N_PAGES
    assert chunks[0]["pos_start"] == 0
    assert chunks[0]["token_count"] == BLOCK_SIZE
    assert chunks[-1]["pos_end"] == NUM_CTX - 1
    for c in chunks[:-1]:
        assert c["logits_last"] is False
    assert chunks[-1]["logits_last"] is True


def test_iter_prefill_execute_chunks_131k_boundaries():
    """Execute chunker matches plan export at 131k scale."""
    if not native_decode_batch_available():
        pytest.skip("native batch ext not built")
    tokens = [0] * NUM_CTX
    exec_chunks = iter_prefill_execute_chunks(tokens, block_size=BLOCK_SIZE)
    assert len(exec_chunks) == N_PAGES
    assert exec_chunks[0] == (tokens[:BLOCK_SIZE], 0, False)
    last_tokens, last_pos, last_logits = exec_chunks[-1]
    assert last_pos == NUM_CTX - BLOCK_SIZE
    assert len(last_tokens) == BLOCK_SIZE
    assert last_logits is True


@pytest.mark.skipif(not native_available(), reason="native ext not built")
def test_page_bind_131k_boundary():
    """8192 PA pages cover positions 0..131071; position 131072 is rejected."""
    from runtime.kv._kv_native import page_bind_clear, page_bind_set
    from runtime.kv.page_bind import validate_token_positions
    from runtime.worker.llama_server import LlamaServerError

    block_ids = list(range(1, N_PAGES + 1))
    page_bind_clear(7)
    page_bind_set(7, BLOCK_SIZE, block_ids)
    try:
        # last valid position in 8192 pages of 16 tokens each
        validate_token_positions(7, NUM_CTX - 1, 1)
        validate_token_positions(7, 0, NUM_CTX)
        with pytest.raises(LlamaServerError, match="token position"):
            validate_token_positions(7, NUM_CTX, 1)
    finally:
        page_bind_clear(7)


@pytest.mark.skipif(not native_available(), reason="native ext not built")
def test_page_bind_set_accepts_8192_pages():
    """Registry cap must fit 131072 ctx at default block_size=16."""
    from runtime.kv._kv_native import page_bind_clear, page_bind_set
    from runtime.kv.tensor_probe import export_page_table

    page_bind_clear(8)
    block_ids = list(range(N_PAGES))
    page_bind_set(8, BLOCK_SIZE, block_ids)
    try:
        rows = export_page_table(8)
        assert len(rows) == N_PAGES
        assert rows[0]["token_start"] == 0
        assert rows[-1]["token_end"] == NUM_CTX - 1
    finally:
        page_bind_clear(8)


@pytest.mark.skipif(not native_available(), reason="native ext not built")
def test_c_page_bind_validate_131k_boundary():
    """C kv_page_bind_validate_range rejects position past 8192 pages."""
    from runtime.kv._kv_native import page_bind_clear, page_bind_set
    from runtime.kv.native_decode_loop import native_decode_loop_available

    if not native_decode_loop_available():
        pytest.skip("linked decode loop not built")

    from runtime.kv._kv_native import decode_loop_prefill

    block_ids = list(range(N_PAGES))
    page_bind_clear(0)
    page_bind_set(0, BLOCK_SIZE, block_ids)
    try:
        with pytest.raises(ValueError, match="KV page bind"):
            decode_loop_prefill(1, [0], 0, BLOCK_SIZE, NUM_CTX)
    finally:
        page_bind_clear(0)
