"""Decode graph cache stub."""

from __future__ import annotations

from runtime.decode_graph_cache import decode_graph_cache
from runtime.decode_graph_policy import reset_decode_graph_epochs


def test_decode_graph_cache_lookup_miss():
    reset_decode_graph_epochs()
    cache = decode_graph_cache()
    assert cache.lookup(2) is None
    assert cache.is_capture_ready() is False


def test_decode_graph_cache_invalidate_changes_key():
    reset_decode_graph_epochs()
    cache = decode_graph_cache()
    key_before = cache.lookup_key(3)
    cache.invalidate_slot(3, reason="test")
    assert cache.lookup_key(3) != key_before


def test_decode_graph_cache_health():
    reset_decode_graph_epochs()
    h = decode_graph_cache().health()
    assert h["stub"] is True
    assert h["capture_ready"] is False
    assert "slot_id:slot_epoch:global_epoch" in h["capture_key_format"]
    assert h["llama_cpp"]["present"] is True
