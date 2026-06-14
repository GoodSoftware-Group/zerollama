"""Tests for Phase 15 v8 page bind (seq-position partial)."""

from __future__ import annotations

import pytest

from runtime.kv.backend import native_available
from runtime.kv.page_bind import (
    page_bind_health,
    page_bind_stats,
    register_request_bind,
    unregister_request_bind,
)


def test_page_bind_health_without_native_ext(monkeypatch):
    monkeypatch.setattr("runtime.kv.page_bind._native_page_bind_available", lambda: False)
    h = page_bind_health(native_ext_available=False)
    assert h["available"] is False
    assert h["status"] == "not_implemented"
    assert "kv_forward_plans" in h["reason"] or "native ext" in h["reason"]
    assert h.get("slots") == []
    assert h.get("bind_level") is None


@pytest.mark.skipif(not native_available(), reason="native ext not built")
def test_page_bind_health_partial_when_native_built():
    h = page_bind_health(native_ext_available=True)
    assert h["available"] is True
    assert h["status"] == "partial"
    assert h["bind_level"] == "seq_position"
    assert h["tensor_pages_bound"] is False
    assert h.get("tensor_bind_ready") is False
    assert "blocker" in h


@pytest.mark.skipif(not native_available(), reason="native ext not built")
def test_page_bind_register_and_resolve():
    from runtime.kv._kv_native import page_bind_clear, page_bind_resolve, page_bind_set

    page_bind_clear(7)
    page_bind_set(7, 16, [3, 9, 12])
    page, bid, off = page_bind_resolve(7, 20)
    assert page == 1
    assert bid == 9
    assert off == 4
    page_bind_clear(7)
    with pytest.raises(KeyError):
        page_bind_resolve(7, 0)


@pytest.mark.skipif(not native_available(), reason="native ext not built")
def test_page_bind_slots_export():
    from runtime.kv._kv_native import page_bind_clear, page_bind_set
    from runtime.kv.page_bind import page_bind_stats

    page_bind_clear(5)
    page_bind_set(5, 16, [1, 2])
    stats = page_bind_stats()
    assert stats["active_binds"] == 1
    slots = stats["slots"]
    assert len(slots) == 1
    assert slots[0]["kv_slot"] == 5
    assert slots[0]["num_pages"] == 2
    assert slots[0]["cell_pages_bound"] is False
    assert slots[0]["tensor_pages_bound"] is False
    page_bind_clear(5)


def test_engine_health_includes_kv_page_bind():
    from pathlib import Path

    from runtime.config import RuntimeConfig
    from runtime.engine import InferenceEngine

    root = Path(__file__).resolve().parents[1]
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=root,
        llama_server_bin=None,
        llama_model=None,
        num_blocks=64,
        block_size=16,
        device_count=1,
    )
    eng = InferenceEngine(cfg)
    h = eng.health()
    assert "kv_page_bind" in h
    pb = h["kv_page_bind"]
    assert "bind_level" in pb or pb.get("status") == "not_implemented"
    if native_available() and pb.get("available"):
        assert "slots" in pb
        assert isinstance(pb["slots"], list)
        assert pb["status"] == "partial"
        assert pb["available"] is True
    else:
        assert pb["status"] == "not_implemented"
    snap = eng.kv_snapshot()
    assert "kv_page_bind" in snap


@pytest.mark.skipif(not native_available(), reason="native ext not built")
def test_page_bind_stats_tensor_pages_bound_bool():
    stats = page_bind_stats()
    assert stats["tensor_pages_bound"] is False
    assert isinstance(stats["tensor_pages_bound"], bool)
    assert isinstance(stats.get("slots"), list)


@pytest.mark.skipif(not native_available(), reason="native ext not built")
def test_validate_token_positions_raises_llama_server_error():
    from runtime.kv._kv_native import page_bind_clear, page_bind_set
    from runtime.kv.page_bind import validate_token_positions
    from runtime.worker.llama_server import LlamaServerError

    page_bind_clear(9)
    page_bind_set(9, 16, [1])
    with pytest.raises(LlamaServerError, match="token position"):
        validate_token_positions(9, 0, 32)
    page_bind_clear(9)


@pytest.mark.skipif(
    not native_available(),
    reason="native ext not built",
)
def test_decode_loop_prefill_c_page_bind_validation():
    """C decode loop rejects overrun before llama_decode (defense in depth).

    Requires the linked ext (ZEROLLAMA_KV_DECODE_LOOP=1); skips when only the
    base ext is built (decode_loop_prefill symbol absent).
    """
    from runtime.kv.native_decode_loop import native_decode_loop_available

    if not native_decode_loop_available():
        pytest.skip("linked decode loop not built (ZEROLLAMA_KV_DECODE_LOOP=1)")

    from runtime.kv._kv_native import decode_loop_prefill, page_bind_clear, page_bind_set

    page_bind_clear(0)
    page_bind_set(0, 16, [1])
    try:
        with pytest.raises(ValueError, match="KV page bind"):
            # pos_start=16 exceeds single registered page — fails before llama_decode
            decode_loop_prefill(1, [0], 0, 16, 16)
    finally:
        page_bind_clear(0)


@pytest.mark.skipif(not native_available(), reason="native ext not built")
def test_scheduler_loop_registers_page_bind_on_admit():
    from runtime.gpu.mutex import InferenceGpuCoordinator
    from runtime.kv._kv_native import page_bind_clear, page_bind_resolve
    from runtime.kv.block_pool import BlockPool
    from runtime.scheduler.loop import SchedulerLoop
    from runtime.scheduler.scheduler import Request, Scheduler

    pool = BlockPool(num_blocks=32, block_size=16, device_id=0)
    sched = Scheduler.for_pools([pool])
    loop = SchedulerLoop(
        scheduler=sched,
        coordinator=InferenceGpuCoordinator(),
        pools=[pool],
        parallel_slots=2,
        assign_llama_slots=True,
    )
    req = Request(
        request_id="loop-bind",
        prompt_tokens=[0] * 20,
        max_tokens=8,
        num_ctx=64,
    )
    sched.add_request(req)
    admitted = loop.tick(max_admit=1)
    assert len(admitted) == 1
    assert page_bind_stats()["active_binds"] == 1
    page, bid, _ = page_bind_resolve(int(admitted[0].kv_slot), 17)
    assert page == 1
    assert bid >= 0
    loop.complete(admitted[0])
    assert page_bind_stats()["active_binds"] == 0
    page_bind_clear(int(admitted[0].kv_slot or 0))


@pytest.mark.skipif(not native_available(), reason="native ext not built")
def test_register_request_bind_from_scheduler_request():
    from runtime.kv._kv_native import page_bind_clear, page_bind_resolve
    from runtime.scheduler.scheduler import Request

    req = Request(
        request_id="t1",
        prompt_tokens=[0] * 10,
        max_tokens=4,
        num_ctx=32,
    )
    req.kv_slot = 2
    req.block_table = type(
        "T",
        (),
        {
            "_tables": [
                type("BT", (), {"block_ids": [5, 8]})(),
            ],
            "num_tokens_capacity": 32,
        },
    )()
    register_request_bind(req, block_size=16)
    assert page_bind_stats()["active_binds"] == 1
    page, bid, _ = page_bind_resolve(2, 17)
    assert page == 1 and bid == 8
    unregister_request_bind(2)
    assert page_bind_stats()["active_binds"] == 0
    page_bind_clear(2)
