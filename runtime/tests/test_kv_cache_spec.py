"""Pluggable KVCacheSpec (vLLM-inspired selective retention)."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from runtime.cache_bridge import DEFAULT_CACHE_TTLS
from runtime.gguf_estimate import GGUF_MAGIC, GGUF_TYPE_UINT32, GgufArchHints
from runtime.kv_cache_spec import (
    KVCacheSpec,
    PrefixCacheRequest,
    classify_gguf_prefix_cache,
    draft_speculative_active,
    effective_prefix_window,
    resolve_kv_cache_spec,
)


def test_standard_spec_allows_cache_with_key():
    spec = KVCacheSpec(
        kind="standard",
        effective_window=8192,
        allow_cache_prompt_base=True,
        allow_disk_persist=True,
        disk_ttl_ms=3600000,
        speculative_draft=False,
    )
    req = PrefixCacheRequest(prompt_cache_key="sess", seq_pos=100, prompt_tokens=50)
    assert spec.cache_prompt_allowed(req) is True
    assert spec.resume_pos(req) == 100


def test_sliding_window_blocks_beyond_window():
    spec = KVCacheSpec(
        kind="sliding_window",
        effective_window=1024,
        allow_cache_prompt_base=True,
        allow_disk_persist=True,
        disk_ttl_ms=300000,
        speculative_draft=False,
    )
    req = PrefixCacheRequest(prompt_cache_key="sess", seq_pos=1024, prompt_tokens=1)
    assert spec.cache_prompt_allowed(req) is False
    assert spec.resume_pos(req) is None


def test_hybrid_spec_ignores_swa_window_for_cache():
    spec = KVCacheSpec(
        kind="hybrid",
        effective_window=2048,
        allow_cache_prompt_base=True,
        allow_disk_persist=True,
        disk_ttl_ms=3600000,
        speculative_draft=False,
    )
    req = PrefixCacheRequest(prompt_cache_key="sess", seq_pos=9000, prompt_tokens=9000)
    assert spec.cache_prompt_allowed(req) is True
    assert spec.resume_pos(req) == 9000


def test_disabled_spec_never_caches():
    spec = KVCacheSpec(
        kind="disabled",
        effective_window=None,
        allow_cache_prompt_base=False,
        allow_disk_persist=False,
        disk_ttl_ms=300000,
        speculative_draft=False,
        notes=("llama_cache_disabled",),
    )
    req = PrefixCacheRequest(prompt_cache_key="sess", seq_pos=10, prompt_tokens=5)
    assert spec.cache_prompt_allowed(req) is False
    assert spec.resume_pos(req) is None


def test_draft_speculative_active():
    assert draft_speculative_active("none") is False
    assert draft_speculative_active("eagle3") is True


def test_resolve_kv_cache_spec_draft_disables(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_LLAMA_CACHE", raising=False)
    spec = resolve_kv_cache_spec(spec_method="mtp")
    assert spec.speculative_draft is True
    assert spec.allow_cache_prompt_base is False
    assert spec.allow_disk_persist is False


def test_resolve_kv_cache_spec_swa_from_gguf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_LLAMA_CACHE", raising=False)
    gguf = tmp_path / "swa.gguf"
    with gguf.open("wb") as f:
        f.write(struct.pack("<II", GGUF_MAGIC, 2))
        kv = [
            (b"llama.block_count", GGUF_TYPE_UINT32, struct.pack("<I", 32)),
            (
                b"llama.attention.sliding_window",
                GGUF_TYPE_UINT32,
                struct.pack("<I", 4096),
            ),
        ]
        f.write(struct.pack("<QQ", 0, len(kv)))
        for key, vtype, payload in kv:
            f.write(struct.pack("<Q", len(key)))
            f.write(key)
            f.write(struct.pack("<I", vtype))
            f.write(payload)
        f.write(struct.pack("<Q", 0))

    spec = resolve_kv_cache_spec(gguf=gguf, num_ctx=8192, spec_method="none")
    assert spec.kind == "sliding_window"
    assert spec.effective_window == 4096
    assert spec.disk_ttl_ms == DEFAULT_CACHE_TTLS["short"]


def test_classify_hybrid_per_layer():
    arch = GgufArchHints(
        scalar={"block_count": 4},
        sliding_window_per_layer=(0, 2048, 0, 2048),
    )
    assert classify_gguf_prefix_cache(arch) == "hybrid"
    assert effective_prefix_window(arch, 8192) == 2048


def test_spec_to_health():
    spec = KVCacheSpec(
        kind="hybrid",
        effective_window=2048,
        allow_cache_prompt_base=True,
        allow_disk_persist=True,
        disk_ttl_ms=3600000,
        speculative_draft=False,
        notes=("swa_kind=hybrid",),
    )
    h = spec.to_health()
    assert h["kind"] == "hybrid"
    assert h["effective_window"] == 2048
    assert "swa_kind=hybrid" in h["notes"]
