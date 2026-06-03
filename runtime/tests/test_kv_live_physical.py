"""Tests for opt-in live kv_physical via multi-seq in-process ctx."""

from __future__ import annotations

import pytest

from runtime.kv.live_physical import (
    effective_parallel_slots,
    kv_live_physical_enabled,
    kv_live_physical_health,
)


def test_kv_live_physical_default_off():
    assert not kv_live_physical_enabled()
    assert (
        effective_parallel_slots([], default=1, backend="inprocess") == 1
    )


def test_kv_live_physical_bumps_inprocess(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_KV_LIVE_PHYSICAL", "1")
    assert effective_parallel_slots([], default=1, backend="inprocess") == 2
    assert effective_parallel_slots([], default=1, backend="subprocess") == 1


def test_kv_live_physical_respects_explicit_np(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_KV_LIVE_PHYSICAL", "1")
    assert (
        effective_parallel_slots(["-np", "1"], default=4, backend="inprocess")
        == 1
    )


def test_kv_live_physical_no_bump_when_yaml_multi(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_KV_LIVE_PHYSICAL", "1")
    assert effective_parallel_slots([], default=4, backend="inprocess") == 4


def test_kv_live_physical_bumps_config_generated_np(monkeypatch):
    """YAML ``-np`` in generated llama args is not an explicit override."""
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_KV_LIVE_PHYSICAL", "1")
    assert (
        effective_parallel_slots(
            ["-sm", "layer", "-mg", "0", "-np", "1"],
            default=1,
            backend="inprocess",
        )
        == 2
    )


def test_kv_live_physical_health_applied(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_KV_LIVE_PHYSICAL", "1")
    h = kv_live_physical_health([], default=1, backend="inprocess")
    assert h["applied"] is True
    assert h["effective"] == 2


def test_engine_health_includes_kv_live_physical(monkeypatch, cfg_root):
    from runtime.config import RuntimeConfig
    from runtime.engine import InferenceEngine

    monkeypatch.setenv("ZEROLLAMA_RUNTIME_KV_LIVE_PHYSICAL", "1")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_LLAMA_BACKEND", "inprocess")
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=cfg_root,
        llama_server_bin=None,
        llama_model=None,
        llama_backend="inprocess",
        num_blocks=64,
        block_size=16,
        device_count=1,
        llama_parallel_slots=1,
    )
    eng = InferenceEngine(cfg)
    h = eng.health()
    live = h.get("kv_live_physical")
    assert live is not None
    assert live["applied"] is True
    assert h.get("kv_inprocess_n_seq_max") == 2
