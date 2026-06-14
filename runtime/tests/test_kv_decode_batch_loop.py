"""Phase 15 v26 — continuous batch decode step (C multi-seq llama_decode)."""

from __future__ import annotations

import pytest

from runtime.kv.backend import native_available
from runtime.kv.decode_plan import kv_continuous_batch_step_plan
from runtime.kv.native_decode_loop import decode_loop_status, native_decode_loop_available


@pytest.mark.skipif(not native_available(), reason="native ext not built")
def test_decode_batch_layout_multi():
    from runtime.kv._kv_native import decode_batch_layout_multi

    layout = decode_batch_layout_multi([10, 20], [0, 1], [5, 12])
    assert layout["token"] == [10, 20]
    assert layout["seq_id"] == [0, 1]
    assert layout["pos"] == [5, 12]
    assert layout["logits"] == [1, 1]


def test_continuous_batch_step_plan_export():
    plan = kv_continuous_batch_step_plan(
        [
            {"token": 1, "seq_id": 0, "pos": 10, "kv_slot": 0},
            {"token": 2, "seq_id": 1, "pos": 20, "kv_slot": 1},
        ],
        block_size=16,
    )
    assert plan["n_batch_rows"] == 2
    assert plan["rows"][0]["page"] == 0
    assert plan["rows"][1]["page"] == 1
    if native_available():
        assert plan.get("batch_layout") is not None


@pytest.mark.skipif(not native_available(), reason="native ext not built")
def test_decode_loop_status_batch_decode_in_c():
    if not native_decode_loop_available():
        pytest.skip("linked decode loop not built")
    st = decode_loop_status()
    assert st.get("batch_decode_in_c") is True


@pytest.mark.skipif(not native_available(), reason="native ext not built")
def test_batch_step_bind_validation_rejects_overrun():
    if not native_decode_loop_available():
        pytest.skip("linked decode loop not built")

    from runtime.kv._kv_native import decode_loop_batch_step, page_bind_clear, page_bind_set

    page_bind_clear(0)
    page_bind_set(0, 16, [1])
    page_bind_clear(1)
    page_bind_set(1, 16, [2])
    try:
        with pytest.raises(ValueError, match="KV page bind"):
            # seq 1 valid at pos 0; seq 0 invalid at pos 16 (one page only)
            decode_loop_batch_step(1, [0, 0], [0, 1], [16, 0])
    finally:
        page_bind_clear(0)
        page_bind_clear(1)


@pytest.mark.skipif(not native_available(), reason="native ext not built")
def test_run_batch_step_maps_bind_error():
    if not native_decode_loop_available():
        pytest.skip("linked decode loop not built")

    from runtime.kv._kv_native import page_bind_clear, page_bind_set
    from runtime.kv.native_decode_loop import run_batch_step
    from runtime.worker.llama_server import LlamaServerError

    page_bind_clear(3)
    page_bind_set(3, 16, [1])
    try:
        with pytest.raises(LlamaServerError, match="KV page bind"):
            run_batch_step(1, [0], [3], [16])
    finally:
        page_bind_clear(3)
