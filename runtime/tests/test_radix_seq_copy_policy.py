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


def _hybrid_spec(*, window: int = 8192) -> KVCacheSpec:
    coord = HybridKVCacheCoordinator(
        kind="hybrid",
        layer_groups=(
            LayerGroupSpec(kind="full", layer_indices=(0, 1), window=None),
            LayerGroupSpec(kind="sliding_window", layer_indices=(2, 3), window=window),
        ),
        num_layers=4,
        num_ctx=8192,
        swa_effective_window=window,
    )
    return KVCacheSpec(
        kind="hybrid",
        effective_window=window,
        allow_cache_prompt_base=True,
        allow_disk_persist=True,
        disk_ttl_ms=300_000,
        speculative_draft=False,
        coordinator=coord,
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
