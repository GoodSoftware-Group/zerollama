"""Hybrid KV cache coordinator (full + SWA layer groups)."""

from __future__ import annotations

import pytest

from runtime.gguf_estimate import GgufArchHints
from runtime.kv.hybrid_kv_coordinator import (
    build_hybrid_kv_coordinator,
    build_layer_groups,
    classify_coordinator_kind,
)
from runtime.kv_cache_spec import resolve_kv_cache_spec


def test_build_layer_groups_hybrid_per_layer():
    arch = GgufArchHints(
        scalar={"block_count": 4},
        sliding_window_per_layer=(0, 2048, 0, 4096),
    )
    groups = build_layer_groups(arch, 8192)
    assert len(groups) == 3
    assert groups[0].kind == "full"
    assert groups[0].layer_indices == (0, 2)
    swa = [g for g in groups if g.kind == "sliding_window"]
    assert len(swa) == 2
    assert swa[0].window == 2048
    assert swa[1].window == 4096


def test_coordinator_hybrid_uses_min_swa_window():
    arch = GgufArchHints(
        scalar={"block_count": 4},
        sliding_window_per_layer=(0, 2048, 0, 4096),
    )
    coord = build_hybrid_kv_coordinator(arch, 8192)
    assert classify_coordinator_kind(coord.layer_groups) == "hybrid"
    assert coord.swa_effective_window == 2048
    assert coord.full_layer_count == 2
    assert coord.swa_layer_count == 2
    assert coord.allows_cache_prompt(seq_pos=2048, prompt_tokens=1) is False
    assert coord.allows_cache_prompt(seq_pos=1500, prompt_tokens=400) is True
    assert coord.coordinated_resume_pos(1500) == 1500
    assert coord.coordinated_resume_pos(3000) is None


def test_coordinator_scalar_sliding_window():
    arch = GgufArchHints(scalar={"block_count": 32, "sliding_window": 4096})
    coord = build_hybrid_kv_coordinator(arch, 8192)
    assert coord.kind == "sliding_window"
    assert coord.swa_effective_window == 4096
    assert len(coord.layer_groups) == 1
    assert coord.layer_groups[0].layer_indices == tuple(range(32))


def test_coordinator_standard_uses_num_ctx():
    arch = GgufArchHints(scalar={"block_count": 32})
    coord = build_hybrid_kv_coordinator(arch, 8192)
    assert coord.kind == "standard"
    assert coord.swa_effective_window == 8192
    assert coord.allows_cache_prompt(seq_pos=7000, prompt_tokens=1000) is True


def test_coordinator_retention_interval(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_PREFIX_CACHE_RETENTION_INTERVAL", "1024")
    arch = GgufArchHints(
        scalar={"block_count": 4},
        sliding_window_per_layer=(0, 2048, 0, 2048),
    )
    coord = build_hybrid_kv_coordinator(arch, 8192)
    assert coord.retention_interval == 1024
    assert coord.allows_cache_prompt(seq_pos=900, prompt_tokens=50) is False
    assert coord.allows_cache_prompt(seq_pos=1024, prompt_tokens=50) is True


def test_resolve_kv_cache_spec_always_attaches_coordinator(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("ZEROLLAMA_LLAMA_CACHE", raising=False)
    spec = resolve_kv_cache_spec(num_ctx=8192, spec_method="none")
    assert spec.coordinator is not None
    assert spec.coordinator.kind == "standard"
    health = spec.to_health()
    assert health["hybrid_coordinator"]["kind"] == "standard"
