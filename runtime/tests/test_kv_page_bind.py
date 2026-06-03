"""Tests for Phase 15 v8 page bind status export."""

from __future__ import annotations

from runtime.kv.page_bind import page_bind_health


def test_page_bind_health_not_implemented():
    h = page_bind_health(native_ext_available=False)
    assert h["available"] is False
    assert h["status"] == "not_implemented"
    assert "kv_forward_plans" in h["reason"]

    h2 = page_bind_health(native_ext_available=True)
    assert h2["native_ext_available"] is True
    assert h2["available"] is False


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
    assert h["kv_page_bind"]["status"] == "not_implemented"
    snap = eng.kv_snapshot()
    assert snap["kv_page_bind"]["available"] is False
