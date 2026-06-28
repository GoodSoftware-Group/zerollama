"""Cross-slot Radix prefix sharing."""

from __future__ import annotations

import pytest

from runtime.kv.prefix_block_pool import (
    build_model_scope,
    get_prefix_block_pool,
    reset_prefix_block_pools_for_tests,
)
from runtime.kv.radix_prefix_share import find_radix_share_plan, radix_prefix_share_enabled
from runtime.kv.lmcache_tier import reset_lmcache_tier_for_tests


def _tokens(n: int, *, base: int = 1000) -> list[int]:
    return [base + i for i in range(n)]


@pytest.fixture(autouse=True)
def _reset():
    reset_prefix_block_pools_for_tests()
    reset_lmcache_tier_for_tests()
    yield
    reset_prefix_block_pools_for_tests()
    reset_lmcache_tier_for_tests()


def test_find_radix_share_plan_cross_slot(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_RADIX_PREFIX_SHARE", "1")
    scope = build_model_scope(model_hash="m1")
    pool = get_prefix_block_pool(model_scope=scope)
    shared = _tokens(1024)
    pool.register_prefix(
        shared,
        scope=scope,
        seq_pos=1024,
        session_key="session-a",
        slot_id=2,
    )
    plan = find_radix_share_plan(
        shared,
        target_slot=5,
        model_hash="m1",
        seq_pos=0,
    )
    assert plan is not None
    assert plan.source_slot == 2
    assert plan.target_slot == 5
    assert plan.copy_tokens == 1024
    assert plan.matched_blocks == 2


def test_find_radix_share_skips_when_target_warm(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_RADIX_PREFIX_SHARE", "1")
    scope = build_model_scope(model_hash="m1")
    pool = get_prefix_block_pool(model_scope=scope)
    tokens = _tokens(512)
    pool.register_prefix(
        tokens,
        scope=scope,
        seq_pos=512,
        session_key="a",
        slot_id=1,
    )
    assert find_radix_share_plan(tokens, target_slot=3, model_hash="m1", seq_pos=512) is None


def test_find_radix_share_same_slot_no_plan(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_RADIX_PREFIX_SHARE", "1")
    scope = build_model_scope(model_hash="m1")
    pool = get_prefix_block_pool(model_scope=scope)
    tokens = _tokens(512)
    pool.register_prefix(
        tokens,
        scope=scope,
        seq_pos=512,
        session_key="a",
        slot_id=3,
    )
    assert find_radix_share_plan(tokens, target_slot=3, model_hash="m1", seq_pos=0) is None


def test_radix_disabled_by_default():
    assert radix_prefix_share_enabled() is False
