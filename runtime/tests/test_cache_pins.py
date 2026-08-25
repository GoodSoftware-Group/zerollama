"""Tests for L3 cache-pin registry."""

from runtime.cache_pins import (
    is_slot_file_pinned,
    pin_ttl_ms_for_file,
    register_cache_pin,
    unregister_cache_pin,
)


def test_cache_pin_extends_slot_ttl(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_LLAMA_CACHE", "1")
    out = register_cache_pin(
        pin_id="cpin_test",
        prompt_cache_key="hermes:agent:x",
        expires_at=None,
    )
    assert out["ok"]
    assert out["slot_ids"]
    # Pick any registered slot filename
    sid = out["slot_ids"][0]
    name = f"slot_{sid}_0.bin"
    assert is_slot_file_pinned(name)
    assert pin_ttl_ms_for_file(name, 1000) > 1000
    unregister_cache_pin(pin_id="cpin_test")
    assert not is_slot_file_pinned(name)
