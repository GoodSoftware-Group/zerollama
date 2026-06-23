"""SWA/hybrid prefix cache policy (vLLM-inspired selective retention)."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from runtime.cache_bridge import DEFAULT_CACHE_TTLS, cache_health
from runtime.gguf_estimate import GGUF_MAGIC, GGUF_TYPE_UINT32, GgufArchHints
from runtime.prefix_cache_policy import (
    PrefixCachePolicy,
    cache_prompt_for_request,
    classify_gguf_prefix_cache,
    draft_speculative_active,
    effective_disk_cache_enabled,
    effective_prefix_window,
    resolve_prefix_cache_policy,
    spec_method_from_hints,
    swa_cache_prompt_allowed,
)
from runtime.llama_args import LlamaServerArgHints


def _write_minimal_gguf(
    path: Path, *, block_count: int = 32, extra_kv: list | None = None
) -> None:
    kv = [
        (b"llama.block_count", GGUF_TYPE_UINT32, struct.pack("<I", block_count)),
    ]
    if extra_kv:
        kv.extend(extra_kv)
    with path.open("wb") as f:
        f.write(struct.pack("<II", GGUF_MAGIC, 2))
        f.write(struct.pack("<QQ", 0, len(kv)))
        for key, vtype, payload in kv:
            f.write(struct.pack("<Q", len(key)))
            f.write(key)
            f.write(struct.pack("<I", vtype))
            f.write(payload)
        f.write(struct.pack("<Q", 0))


def test_classify_standard():
    arch = GgufArchHints(scalar={"block_count": 32})
    assert classify_gguf_prefix_cache(arch) == "standard"


def test_classify_scalar_sliding_window():
    arch = GgufArchHints(scalar={"block_count": 32, "sliding_window": 4096})
    assert classify_gguf_prefix_cache(arch) == "sliding_window"


def test_classify_hybrid_per_layer():
    arch = GgufArchHints(
        scalar={"block_count": 4},
        sliding_window_per_layer=(0, 2048, 0, 2048),
    )
    assert classify_gguf_prefix_cache(arch) == "hybrid"


def test_effective_prefix_window_swa():
    arch = GgufArchHints(scalar={"block_count": 32, "sliding_window": 4096})
    assert effective_prefix_window(arch, 8192) == 4096


def test_effective_prefix_window_per_layer_ignores_scalar_duplicate():
    """When per-layer SWA is authoritative, scalar sliding_window is not double-counted."""
    arch = GgufArchHints(
        scalar={"block_count": 4, "sliding_window": 99999},
        sliding_window_per_layer=(0, 2048, 0, 4096),
    )
    assert effective_prefix_window(arch, 8192) == 2048


def test_spec_method_from_hints():
    assert spec_method_from_hints(LlamaServerArgHints()) == "none"
    assert spec_method_from_hints(LlamaServerArgHints(spec_type="eagle3")) == "eagle3"


def test_draft_speculative_active():
    assert draft_speculative_active("none") is False
    assert draft_speculative_active("ngram") is False
    assert draft_speculative_active("eagle3") is True
    assert draft_speculative_active("mtp") is True


def test_resolve_policy_disables_disk_for_draft_spec(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_LLAMA_CACHE", raising=False)
    monkeypatch.delenv("ZEROLLAMA_LLAMA_CACHE_DISK", raising=False)
    policy = resolve_prefix_cache_policy(spec_method="eagle3")
    assert policy.allow_cache_prompt is False
    assert policy.allow_disk_persist is False
    assert "disk_disabled_draft_speculative" in policy.notes
    assert "cache_prompt_disabled_draft_speculative" in policy.notes


def test_swa_cache_prompt_allowed_hybrid_always():
    policy = PrefixCachePolicy(
        kind="hybrid",
        allow_cache_prompt=True,
        allow_disk_persist=True,
        effective_window=2048,
        disk_ttl_ms=3600000,
        speculative_draft=False,
    )
    assert swa_cache_prompt_allowed(policy, seq_pos=5000, prompt_tokens=9000) is True


def test_swa_cache_prompt_blocks_beyond_window():
    policy = PrefixCachePolicy(
        kind="sliding_window",
        allow_cache_prompt=True,
        allow_disk_persist=True,
        effective_window=4096,
        disk_ttl_ms=300000,
        speculative_draft=False,
    )
    assert swa_cache_prompt_allowed(policy, seq_pos=0, prompt_tokens=1000) is True
    assert swa_cache_prompt_allowed(policy, seq_pos=4096, prompt_tokens=10) is False
    assert swa_cache_prompt_allowed(policy, seq_pos=0, prompt_tokens=5000) is False
    assert swa_cache_prompt_allowed(policy, seq_pos=3000, prompt_tokens=2000) is False


def test_cache_prompt_for_request_swa_enforcement(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_LLAMA_CACHE", raising=False)
    policy = PrefixCachePolicy(
        kind="sliding_window",
        allow_cache_prompt=True,
        allow_disk_persist=True,
        effective_window=1024,
        disk_ttl_ms=300000,
        speculative_draft=False,
    )
    assert cache_prompt_for_request("key", policy, seq_pos=0, prompt_tokens=512) is True
    assert cache_prompt_for_request("key", policy, seq_pos=1024, prompt_tokens=1) is False


def test_effective_disk_cache_respects_policy_and_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_LLAMA_CACHE_DISK", raising=False)
    draft = resolve_prefix_cache_policy(spec_method="mtp")
    assert effective_disk_cache_enabled(draft) is False
    std = resolve_prefix_cache_policy(spec_method="none")
    assert effective_disk_cache_enabled(std) is True
    monkeypatch.setenv("ZEROLLAMA_LLAMA_CACHE_DISK", "0")
    assert effective_disk_cache_enabled(std) is False


def test_resolve_policy_from_gguf_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_LLAMA_CACHE", raising=False)
    gguf = tmp_path / "swa.gguf"
    _write_minimal_gguf(
        gguf,
        extra_kv=[
            (b"llama.attention.sliding_window", GGUF_TYPE_UINT32, struct.pack("<I", 4096)),
        ],
    )
    policy = resolve_prefix_cache_policy(gguf=gguf, num_ctx=8192, spec_method="none")
    assert policy.kind == "sliding_window"
    assert policy.effective_window == 4096
    assert policy.disk_ttl_ms == DEFAULT_CACHE_TTLS["short"]


def test_draft_spec_disables_disk_in_cache_health(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_LLAMA_CACHE", raising=False)
    monkeypatch.delenv("ZEROLLAMA_LLAMA_CACHE_DISK", raising=False)
    gguf = tmp_path / "m.gguf"
    _write_minimal_gguf(gguf)
    h = cache_health(gguf, [], spec_method="eagle3")
    assert h["policy"]["speculative_draft"] is True
    assert h["inprocess_disk_cache"] is False


def test_cache_health_disk_off_until_model_on_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_LLAMA_CACHE", raising=False)
    monkeypatch.delenv("ZEROLLAMA_LLAMA_CACHE_DISK", raising=False)
    missing = tmp_path / "not-yet-pulled.gguf"
    h = cache_health(missing, [])
    assert h["model_loaded"] is False
    assert h["inprocess_disk_cache"] is False
