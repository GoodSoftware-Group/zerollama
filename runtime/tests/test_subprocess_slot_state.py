"""Subprocess slot position tracking for L3 SWA policy."""

from __future__ import annotations

from runtime.subprocess_slot_state import (
    SubprocessSlotState,
    seq_pos_from_llama_result,
    seq_pos_from_slot_entry,
)


def test_seq_pos_from_llama_timings():
    result = {
        "content": "hi",
        "timings": {"cache_n": 100, "prompt_n": 50, "predicted_n": 32},
    }
    assert seq_pos_from_llama_result(result) == 182


def test_seq_pos_missing_timings():
    assert seq_pos_from_llama_result({"content": "x"}) is None


def test_subprocess_slot_state_tracks_per_slot():
    state = SubprocessSlotState()
    assert state.seq_pos(2) is None
    state.record_completion(
        2,
        {"timings": {"cache_n": 0, "prompt_n": 3000, "predicted_n": 32}},
    )
    assert state.seq_pos(2) == 3032
    state.record_completion(
        2,
        {"timings": {"cache_n": 3000, "prompt_n": 500, "predicted_n": 16}},
    )
    assert state.seq_pos(2) == 3516
    assert state.snapshot() == {"2": 3516}


def test_seq_pos_from_slot_entry():
    entry = {
        "id": 1,
        "n_prompt_tokens": 84,
        "next_token": {"n_decoded": 12},
    }
    assert seq_pos_from_slot_entry(entry) == 96


def test_merge_slots_backfills_empty_cache():
    state = SubprocessSlotState()
    state.merge_slots(
        [
            {"id": 0, "is_processing": False},
            {"id": 1, "n_prompt_tokens": 2500, "next_token": {"n_decoded": 32}},
        ]
    )
    assert state.seq_pos(0) is None
    assert state.seq_pos(1) == 2532


def test_seq_pos_with_fallback_fetches_slots(monkeypatch):
    state = SubprocessSlotState()

    def _fake_fetch(base_url: str, *, timeout: float = 2.0):
        assert base_url == "http://127.0.0.1:8082"
        return [{"id": 3, "n_prompt_tokens": 1200, "next_token": {"n_decoded": 8}}]

    monkeypatch.setattr(
        "runtime.subprocess_slot_state.fetch_llama_server_slots", _fake_fetch
    )
    assert state.seq_pos_with_fallback(3, "http://127.0.0.1:8082") == 1208
    assert state.seq_pos(3) == 1208
    # Second call uses local cache — no extra fetch.
    assert state.seq_pos_with_fallback(3, "http://127.0.0.1:8082") == 1208


def test_swa_blocks_turn2_when_pos_plus_prompt_exceeds_window():
    from runtime.prefix_cache_policy import (
        PrefixCachePolicy,
        cache_prompt_for_request,
    )

    policy = PrefixCachePolicy(
        kind="sliding_window",
        allow_cache_prompt=True,
        allow_disk_persist=True,
        effective_window=4096,
        disk_ttl_ms=300000,
        speculative_draft=False,
    )
    # Turn 1: fresh slot, prompt fits window.
    assert cache_prompt_for_request("key", policy, seq_pos=0, prompt_tokens=3000) is True
    # Turn 2: slot retained 3032 tokens; full prompt 3500 → pos + n_prompt > window.
    assert (
        cache_prompt_for_request("key", policy, seq_pos=3032, prompt_tokens=3500)
        is False
    )
