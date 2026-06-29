"""Engine radix share guards."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from runtime.kv_cache_spec import KVCacheSpec
from runtime.prefix_cache_policy import PrefixCachePolicy


def _hybrid_policy(*, window: int = 8192) -> PrefixCachePolicy:
    return PrefixCachePolicy(
        kind="hybrid",
        allow_cache_prompt=True,
        allow_disk_persist=True,
        effective_window=window,
        disk_ttl_ms=300000,
        speculative_draft=False,
    )


def test_apply_radix_hybrid_allowed_within_window(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_RADIX_PREFIX_SHARE", "1")
    monkeypatch.setenv("ZEROLLAMA_PREFIX_BLOCK_POOL", "1")
    monkeypatch.setenv("ZEROLLAMA_RADIX_HYBRID_SEQ_COPY", "1")

    from runtime.engine import InferenceEngine
    from runtime.scheduler.scheduler import Request

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
    engine._inprocess_ctx_for_health = MagicMock(return_value=None)
    engine._subprocess_base_url = MagicMock(return_value="http://127.0.0.1:8082")
    engine._is_kv_slot_busy = MagicMock(return_value=False)
    engine._resolved_llama_backend = MagicMock(return_value=MagicMock())
    engine.config = MagicMock(speculative=MagicMock(method="none"))
    engine._model_hash_for_cache = MagicMock(return_value="mh")

    plan = MagicMock(
        source_slot=0,
        target_slot=2,
        copy_tokens=128,
        matched_blocks=1,
        tail_block_hash=None,
        warm_catchup=False,
        target_seq_pos_before=0,
    )

    with patch(
        "runtime.kv.radix_prefix_share.find_radix_share_plan",
        return_value=plan,
    ), patch(
        "runtime.kv.radix_seq_copy.execute_radix_share_plan",
        return_value=True,
    ), patch(
        "runtime.decode_graph_policy.bump_decode_graph_epoch",
    ), patch(
        "runtime.prefix_cache_trace.record_radix_share",
    ):
        allow, resume, trace = engine._apply_radix_prefix_share(
            req,
            _hybrid_policy(),
            allow=True,
            resume_pos=0,
        )

    assert allow is True
    assert resume == 128
    assert trace is not None
    assert trace.get("copy_tokens") == 128


def test_apply_radix_hybrid_skipped_beyond_window(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_RADIX_PREFIX_SHARE", "1")
    monkeypatch.setenv("ZEROLLAMA_PREFIX_BLOCK_POOL", "1")
    monkeypatch.setenv("ZEROLLAMA_RADIX_HYBRID_SEQ_COPY", "1")

    from runtime.engine import InferenceEngine
    from runtime.scheduler.scheduler import Request

    req = Request(
        request_id="r1",
        prompt_tokens=list(range(2048)),
        max_tokens=8,
        prompt_cache_key="k",
        kv_slot=2,
        slot_pinned=True,
    )

    engine = MagicMock(spec=InferenceEngine)
    engine._apply_radix_prefix_share = InferenceEngine._apply_radix_prefix_share.__get__(
        engine, InferenceEngine
    )
    engine._model_hash_for_cache = MagicMock(return_value="mh")

    plan = MagicMock(
        source_slot=0,
        target_slot=2,
        copy_tokens=2048,
        matched_blocks=4,
        tail_block_hash=None,
        warm_catchup=False,
        target_seq_pos_before=0,
    )

    with patch(
        "runtime.kv.radix_prefix_share.find_radix_share_plan",
        return_value=plan,
    ):
        allow, resume, trace = engine._apply_radix_prefix_share(
            req,
            _hybrid_policy(window=512),
            allow=True,
            resume_pos=0,
        )

    assert allow is True
    assert resume == 0
    assert trace == {"skipped": "hybrid_prefix_exceeds_swa_window"}
