"""Subprocess L3 prefix cache policy (vLLM drop_eagle_block fallback)."""

from __future__ import annotations

from runtime.kv_cache_spec import KVCacheSpec
from runtime.prefix_cache_policy import (
    PrefixCachePolicy,
    decode_graph_invalidation_reason,
    prefix_cache_decision,
)


def _draft_policy() -> PrefixCachePolicy:
    spec = KVCacheSpec(
        kind="standard",
        effective_window=8192,
        allow_cache_prompt_base=True,
        allow_disk_persist=False,
        disk_ttl_ms=3600000,
        speculative_draft=True,
        drop_last_block_on_resume=True,
    )
    return PrefixCachePolicy.from_spec(spec)


def test_prefix_cache_decision_inprocess_keeps_draft_drop():
    allow, resume, deny = prefix_cache_decision(
        "sess",
        _draft_policy(),
        seq_pos=4096,
        prompt_tokens=32,
        subprocess=False,
    )
    assert deny is None
    assert allow is True
    assert resume == 3584


def test_prefix_cache_decision_subprocess_disables_draft_drop():
    allow, resume, deny = prefix_cache_decision(
        "sess",
        _draft_policy(),
        seq_pos=4096,
        prompt_tokens=32,
        subprocess=True,
    )
    assert allow is False
    assert resume is None
    assert deny == "subprocess_drop_last_block_unsupported"


def test_decode_graph_invalidation_reason_subprocess_draft():
    reason = decode_graph_invalidation_reason(
        allow=False,
        resume=None,
        seq_pos=4096,
        slot_pinned=True,
        deny_reason="subprocess_drop_last_block_unsupported",
    )
    assert reason == "subprocess_drop_last_block"


def test_decode_graph_invalidation_reason_swa_block():
    reason = decode_graph_invalidation_reason(
        allow=False,
        resume=None,
        seq_pos=2048,
        slot_pinned=True,
    )
    assert reason == "cache_prompt_disabled"
