"""Phase 15 v4: llama seq position vs PA reserve."""

from __future__ import annotations

import pytest

from runtime.kv.block_pool import BlockPool, SequenceBlockTable
from runtime.kv.physical import (
    SequenceKvUsage,
    pa_llama_alignment,
    physical_strict_enabled,
)
from runtime.kv.multi_pool import MultiDeviceBlockTable
from runtime.scheduler.scheduler import Request
from runtime.worker.llama_server import LlamaServerError


def test_sequence_kv_usage_token_cells():
    assert SequenceKvUsage(0, 0, 15).token_cells == 16
    assert SequenceKvUsage(0, -1, -1).token_cells == 0
    assert SequenceKvUsage(1, 2, 10).token_cells == 9


def test_pa_llama_alignment_ok():
    pool = BlockPool(num_blocks=16, block_size=16, device_id=0)
    req = Request("a", [0] * 10, max_tokens=8)
    req.block_table = MultiDeviceBlockTable(request_id="a", pools=[pool])
    req.block_table.ensure_capacity(64)
    usage = SequenceKvUsage(seq_id=0, pos_min=0, pos_max=19)
    row = pa_llama_alignment(req, usage, block_size=16)
    assert row["aligned"] is True
    assert row["pages_fit"] is True
    assert row["llama_token_cells"] == 20


def test_pa_llama_alignment_overflow_strict(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_KV_PHYSICAL_STRICT", "1")
    pool = BlockPool(num_blocks=2, block_size=16, device_id=0)
    req = Request("b", [0] * 5, max_tokens=4, kv_slot=1)
    req.block_table = MultiDeviceBlockTable(request_id="b", pools=[pool])
    req.block_table.ensure_capacity(32)
    usage = SequenceKvUsage(seq_id=0, pos_min=0, pos_max=40)
    row = pa_llama_alignment(req, usage, block_size=16)
    assert row["aligned"] is False
    from runtime.kv.physical import verify_after_decode

    with pytest.raises(LlamaServerError, match="request_id=b.*kv_slot=1"):
        verify_after_decode(req, usage, block_size=16, at="test")


def test_kv_physical_health_pa_only_note():
    from runtime.kv.physical import kv_physical_health_pa_only

    pool = BlockPool(num_blocks=8, block_size=16, device_id=0)
    req = Request("z", [0], max_tokens=4)
    req.block_table = MultiDeviceBlockTable(request_id="z", pools=[pool])
    req.block_table.ensure_capacity(32)
    snap = kv_physical_health_pa_only([req], block_size=16)
    assert snap["note"]
    assert snap["running"][0]["live_seq_positions"] is False
    assert snap["running"][0]["llama_tracked"] is False


def test_recent_alignments_ring_mismatch_only():
    from runtime.kv.physical import (
        SequenceKvUsage,
        clear_recent_alignments_for_tests,
        recent_alignments,
        verify_after_decode,
    )
    from runtime.kv.multi_pool import MultiDeviceBlockTable

    clear_recent_alignments_for_tests()
    pool = BlockPool(num_blocks=4, block_size=16, device_id=0)
    req = Request("ok", [0] * 4, max_tokens=4)
    req.block_table = MultiDeviceBlockTable(request_id="ok", pools=[pool])
    req.block_table.ensure_capacity(32)
    verify_after_decode(
        req, SequenceKvUsage(0, 0, 5), block_size=16, at="ok"
    )
    req2 = Request("bad", [0] * 4, max_tokens=4)
    req2.block_table = MultiDeviceBlockTable(request_id="bad", pools=[pool])
    req2.block_table.ensure_capacity(32)
    verify_after_decode(
        req2, SequenceKvUsage(0, 0, 200), block_size=16, at="bad"
    )
    rows = recent_alignments()
    assert len(rows) == 1
    assert rows[0]["request_id"] == "bad"


def test_native_scheduler_tick_optional():
    pytest.importorskip("runtime.kv._kv_native", reason="native extension not built")
    from runtime.kv._kv_native import scheduler_tick
    from runtime.kv.native_tick import record_scheduler_tick

    a = int(scheduler_tick())
    b = record_scheduler_tick()
    assert b == a + 1
