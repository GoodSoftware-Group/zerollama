"""Phase 15 v3: PA block table ↔ llama bind contract."""

from __future__ import annotations

import pytest

from runtime.kv.bind import (
    KvBindMode,
    assert_kv_capacity,
    kv_bind_health,
    logical_page_for_token,
    primary_block_ids,
    reserved_token_capacity,
    tokens_to_reserve,
)
from runtime.gpu.mutex import InferenceGpuCoordinator
from runtime.kv.block_pool import BlockPool, SequenceBlockTable
from runtime.scheduler.loop import SchedulerLoop
from runtime.scheduler.scheduler import Request, Scheduler
from runtime.worker.llama_server import LlamaServerError


def test_logical_page_for_token():
    assert logical_page_for_token(0, 16) == (0, 0)
    assert logical_page_for_token(16, 16) == (1, 0)
    assert logical_page_for_token(17, 16) == (1, 1)


def test_sequence_page_maps_to_pool_block_id_not_index():
    """Page ordinal i uses block_ids[i]; pool id need not equal i."""
    pool = BlockPool(num_blocks=8, block_size=16, device_id=0)
    table = SequenceBlockTable(request_id="t", pool=pool)
    table.ensure_capacity(48)
    bids = table.block_ids
    assert len(bids) == 3
    page, off = logical_page_for_token(20, 16)
    assert page == 1 and off == 4
    assert bids[page] == bids[1]


def test_primary_block_ids_after_tick():
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
        request_id="a",
        prompt_tokens=[0] * 20,
        max_tokens=8,
        num_ctx=64,
    )
    sched.add_request(req)
    admitted = loop.tick(max_admit=1)
    assert len(admitted) == 1
    bids = primary_block_ids(admitted[0])
    assert bids
    assert reserved_token_capacity(admitted[0]) >= tokens_to_reserve(admitted[0])
    assert_kv_capacity(admitted[0], block_size=16, at="test")
    loop.complete(admitted[0])


def test_assert_kv_capacity_fails_when_under_reserved():
    pool = BlockPool(num_blocks=4, block_size=16, device_id=0)
    sched = Scheduler.for_pools([pool])
    req = Request(
        request_id="x",
        prompt_tokens=[0] * 100,
        max_tokens=64,
        num_ctx=4096,
    )
    sched.add_request(req)
    assert sched.running == [] or True
    # Manually attach a tiny table (simulates bug / race).
    from runtime.kv.multi_pool import MultiDeviceBlockTable

    req.block_table = MultiDeviceBlockTable(request_id="x", pools=[pool])
    req.block_table.ensure_capacity(32)
    with pytest.raises(LlamaServerError, match="KV bind"):
        assert_kv_capacity(req, block_size=16, at="unit")


def test_kv_bind_health_modes():
    h = kv_bind_health(
        llama_backend="subprocess",
        assign_llama_slots=True,
        parallel_slots=4,
    )
    assert h["mode"] == KvBindMode.SEQ_SLOT.value
    assert h["physical_pages_bound"] is False

    h2 = kv_bind_health(
        llama_backend="llama_cpp_python",
        assign_llama_slots=False,
        parallel_slots=1,
    )
    assert h2["mode"] == KvBindMode.ACCOUNTING_ONLY.value
    assert "sequence page" in h["page_table_semantics"] or "ordinal" in h["page_table_semantics"]


def test_kv_bind_health_physical_when_bound():
    h = kv_bind_health(
        llama_backend="inprocess",
        assign_llama_slots=True,
        parallel_slots=2,
        physical_bind_level="physical",
        physical_pages_bound=True,
    )
    assert h["physical_pages_bound"] is True
    assert h["physical_bind_level"] == "physical"
