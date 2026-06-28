"""Engine radix share guards."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from runtime.kv_cache_spec import KVCacheSpec
from runtime.prefix_cache_policy import PrefixCachePolicy


def test_apply_radix_skips_hybrid_memory(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_RADIX_PREFIX_SHARE", "1")
    monkeypatch.setenv("ZEROLLAMA_PREFIX_BLOCK_POOL", "1")

    from runtime.engine import InferenceEngine
    from runtime.scheduler.scheduler import Request

    policy = PrefixCachePolicy(
        kind="hybrid",
        allow_cache_prompt=True,
        allow_disk_persist=True,
        effective_window=8192,
        disk_ttl_ms=300000,
        speculative_draft=False,
    )
    req = Request(
        request_id="r1",
        prompt_tokens=list(range(128)),
        max_tokens=8,
        prompt_cache_key="k",
        kv_slot=2,
        slot_pinned=True,
    )

    engine = MagicMock(spec=InferenceEngine)
    engine._apply_radix_prefix_share = InferenceEngine._apply_radix_prefix_share.__get__(
        engine, InferenceEngine
    )

    with patch(
        "runtime.kv.radix_prefix_share.find_radix_share_plan",
        return_value=MagicMock(
            source_slot=0,
            target_slot=2,
            copy_tokens=128,
            matched_blocks=2,
            tail_block_hash=None,
        ),
    ):
        allow, resume, trace = engine._apply_radix_prefix_share(
            req,
            policy,
            allow=True,
            resume_pos=0,
        )

    assert allow is True
    assert resume == 0
    assert trace == {"skipped": "hybrid_memory_seq_cp_unsupported"}
