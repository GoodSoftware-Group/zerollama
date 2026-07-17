"""Radix seq-copy policy for hybrid GGUF layouts."""

from __future__ import annotations

import pytest

from runtime.kv.hybrid_kv_coordinator import HybridKVCacheCoordinator, LayerGroupSpec
from runtime.kv.radix_prefix_share import RadixSharePlan
from runtime.kv.radix_seq_copy_policy import radix_seq_copy_allowed
from runtime.kv_cache_spec import KVCacheSpec


def _plan(copy_tokens: int = 128, *, target_seq_pos_before: int = 0) -> RadixSharePlan:
    return RadixSharePlan(
        source_slot=0,
        target_slot=2,
        copy_tokens=copy_tokens,
        matched_blocks=1,
        tail_block_hash=None,
        target_seq_pos_before=target_seq_pos_before,
        warm_catchup=target_seq_pos_before > 0,
    )


def _hybrid_spec(*, window: int = 8192, retention: int | None = None) -> KVCacheSpec:
    coord = HybridKVCacheCoordinator(
        kind="hybrid",
        layer_groups=(
            LayerGroupSpec(kind="full", layer_indices=(0, 1), window=None),
            LayerGroupSpec(kind="sliding_window", layer_indices=(2, 3), window=window),
        ),
        num_layers=4,
        num_ctx=8192,
        swa_effective_window=window,
        retention_interval=retention,
    )
    return KVCacheSpec(
        kind="hybrid",
        effective_window=window,
        allow_cache_prompt_base=True,
        allow_disk_persist=True,
        disk_ttl_ms=300_000,
        speculative_draft=False,
        coordinator=coord,
        retention_interval=retention,
    )


def test_standard_kind_allowed():
    spec = KVCacheSpec(
        kind="standard",
        effective_window=8192,
        allow_cache_prompt_base=True,
        allow_disk_persist=True,
        disk_ttl_ms=300_000,
        speculative_draft=False,
    )
    ok, reason = radix_seq_copy_allowed(spec, _plan(512))
    assert ok is True
    assert reason is None


def test_hybrid_allowed_within_swa_window(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_RADIX_HYBRID_SEQ_COPY", "1")
    ok, reason = radix_seq_copy_allowed(_hybrid_spec(window=8192), _plan(512))
    assert ok is True
    assert reason is None


def test_hybrid_denied_beyond_swa_window(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_RADIX_HYBRID_SEQ_COPY", "1")
    ok, reason = radix_seq_copy_allowed(_hybrid_spec(window=512), _plan(1024))
    assert ok is False
    assert reason == "hybrid_prefix_exceeds_swa_window"


def test_hybrid_denied_when_env_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_RADIX_HYBRID_SEQ_COPY", "0")
    ok, reason = radix_seq_copy_allowed(_hybrid_spec(), _plan(128))
    assert ok is False
    assert reason == "hybrid_seq_copy_disabled"


def test_hybrid_cold_allowed_under_retention(monkeypatch: pytest.MonkeyPatch):
    """Marconi analog: cold shared-prefix seed must not die on retention interval."""
    monkeypatch.setenv("ZEROLLAMA_RADIX_HYBRID_SEQ_COPY", "1")
    monkeypatch.setenv("ZEROLLAMA_PREFIX_CACHE_RETENTION_INTERVAL", "1024")
    # Odd length would fail mid-seq retention; cold pos=0 must still pass.
    ok, reason = radix_seq_copy_allowed(
        _hybrid_spec(window=8192, retention=1024),
        _plan(777),
    )
    assert ok is True
    assert reason is None


def test_hybrid_warm_catchup_ignores_retention(monkeypatch: pytest.MonkeyPatch):
    """Retention is for same-slot resume; warm Radix copies dense donor KV."""
    monkeypatch.setenv("ZEROLLAMA_RADIX_HYBRID_SEQ_COPY", "1")
    ok, reason = radix_seq_copy_allowed(
        _hybrid_spec(window=8192, retention=1024),
        _plan(2048, target_seq_pos_before=800),
    )
    assert ok is True
    assert reason is None


def test_hybrid_warm_denied_past_swa_window(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_RADIX_HYBRID_SEQ_COPY", "1")
    ok, reason = radix_seq_copy_allowed(
        _hybrid_spec(window=512),
        _plan(256, target_seq_pos_before=600),
    )
    assert ok is False
    assert reason == "hybrid_target_past_swa_window"
