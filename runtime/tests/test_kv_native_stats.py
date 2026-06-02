"""Phase 15 v7: read-only native kv_stats()."""

from __future__ import annotations

import pytest

from runtime.kv.native_stats import native_kv_available, native_kv_stats
from runtime.kv.native_tick import record_scheduler_tick
from runtime.kv.native_decode import record_decode_step


@pytest.mark.skipif(not native_kv_available(), reason="native ext not built")
def test_native_kv_stats_reads_counters():
    record_scheduler_tick()
    record_decode_step(2)
    stats = native_kv_stats()
    assert stats is not None
    assert stats["scheduler_tick"] >= 1
    assert stats["decode_steps"] >= 2


def test_native_kv_stats_none_without_ext():
    if native_kv_available():
        pytest.skip("native ext present")
    assert native_kv_stats() is None
