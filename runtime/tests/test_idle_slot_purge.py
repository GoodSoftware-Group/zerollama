"""Phase 15 v57 — in-process idle-slot purge under kv_unified."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from runtime.kv.idle_slot_purge import (
    idle_slot_purge_enabled,
    idle_slot_purge_health,
    llama_decode_with_idle_purge,
    reset_idle_purge_stats_for_tests,
    try_clear_idle_slot,
)
from runtime.kv.physical import SequenceKvUsage


def test_idle_purge_disabled_without_unified(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_KV_UNIFIED_IDLE_PURGE", raising=False)
    assert idle_slot_purge_enabled(kv_unified=False) is False
    assert idle_slot_purge_enabled(kv_unified=True) is True
    monkeypatch.setenv("ZEROLLAMA_KV_UNIFIED_IDLE_PURGE", "0")
    assert idle_slot_purge_enabled(kv_unified=True) is False


def test_try_clear_idle_slot_picks_largest(monkeypatch: pytest.MonkeyPatch):
    reset_idle_purge_stats_for_tests()
    cleared: list[int] = []

    def fake_usage(lib, ctx, sid):
        # seq 0 empty, seq 1=10 cells, seq 2=50 cells; keep=0 → purge 2
        cells = {0: 0, 1: 10, 2: 50}.get(sid, 0)
        if cells <= 0:
            return SequenceKvUsage(seq_id=sid, pos_min=-1, pos_max=-1)
        return SequenceKvUsage(seq_id=sid, pos_min=0, pos_max=cells - 1)

    monkeypatch.setattr(
        "runtime.kv.physical.usage_from_libllama", fake_usage
    )
    sid = try_clear_idle_slot(
        MagicMock(),
        MagicMock(),
        keep_seq=0,
        n_seq_max=3,
        clear_fn=lambda lib, ctx, s: cleared.append(s),
    )
    assert sid == 2
    assert cleared == [2]
    health = idle_slot_purge_health()
    assert health["purged_total"] == 1
    assert health["last_purged_seq"] == 2


def test_try_clear_idle_slot_skips_keep(monkeypatch: pytest.MonkeyPatch):
    reset_idle_purge_stats_for_tests()

    def fake_usage(lib, ctx, sid):
        return SequenceKvUsage(seq_id=sid, pos_min=0, pos_max=99)

    monkeypatch.setattr(
        "runtime.kv.physical.usage_from_libllama", fake_usage
    )
    sid = try_clear_idle_slot(
        MagicMock(),
        MagicMock(),
        keep_seq=1,
        n_seq_max=2,
        clear_fn=lambda *a: None,
    )
    assert sid == 0


def test_llama_decode_retries_after_purge(monkeypatch: pytest.MonkeyPatch):
    reset_idle_purge_stats_for_tests()
    lib = MagicMock()
    lib.llama_decode.side_effect = [1, 0]  # fail then ok
    purged: list[int] = []

    monkeypatch.setattr(
        "runtime.kv.idle_slot_purge.try_clear_idle_slot",
        lambda *a, **k: 2,
    )
    rc = llama_decode_with_idle_purge(
        lib,
        MagicMock(),
        MagicMock(),
        keep_seq=0,
        n_seq_max=4,
        kv_unified=True,
        on_purge=purged.append,
    )
    assert rc == 0
    assert purged == [2]
    assert lib.llama_decode.call_count == 2


def test_llama_decode_no_retry_when_purge_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_KV_UNIFIED_IDLE_PURGE", "0")
    lib = MagicMock()
    lib.llama_decode.return_value = 1
    rc = llama_decode_with_idle_purge(
        lib,
        MagicMock(),
        MagicMock(),
        keep_seq=0,
        n_seq_max=4,
        kv_unified=True,
    )
    assert rc == 1
    assert lib.llama_decode.call_count == 1
