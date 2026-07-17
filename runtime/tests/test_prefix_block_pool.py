"""Hash-chained prefix block pool + optional LMCache tier."""

from __future__ import annotations

import pytest

from runtime.env import reset_runtime_env_for_tests
from runtime.kv.lmcache_tier import (
    LMCacheBlockRecord,
    LMCacheTierStore,
    lmcache_tier,
    reset_lmcache_tier_for_tests,
)
from runtime.kv.prefix_block_hash import iter_prefix_blocks, model_scope_key, prefix_block_hash
from runtime.kv.prefix_block_pool import (
    build_model_scope,
    get_prefix_block_pool,
    prefix_block_pool_enabled,
    reset_prefix_block_pools_for_tests,
)
from runtime.prefix_cache_policy import prefix_cache_decision, resolve_prefix_cache_policy


def _tokens(n: int, *, base: int = 100) -> list[int]:
    return [base + i for i in range(n)]


@pytest.fixture(autouse=True)
def _reset_pools():
    reset_runtime_env_for_tests()
    reset_prefix_block_pools_for_tests()
    reset_lmcache_tier_for_tests()
    yield
    reset_runtime_env_for_tests()
    reset_prefix_block_pools_for_tests()
    reset_lmcache_tier_for_tests()


def test_prefix_block_hash_chain_is_stable():
    scope = model_scope_key(model_hash="abc123")
    tokens = _tokens(16)
    blocks = list(iter_prefix_blocks(tokens, block_size=8, scope=scope))
    assert len(blocks) == 2
    parent = "0" * 64
    h0 = prefix_block_hash(scope=scope, parent_hash=parent, block_index=0, tokens=tokens[:8])
    h1 = prefix_block_hash(scope=scope, parent_hash=h0, block_index=1, tokens=tokens[8:16])
    assert blocks[0][4] == h0
    assert blocks[1][4] == h1


def test_register_and_lookup_prefix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_PREFIX_BLOCK_POOL", "1")
    scope = build_model_scope(model_hash="model1")
    pool = get_prefix_block_pool(model_scope=scope)
    tokens = _tokens(1024)
    pool.register_prefix(
        tokens,
        scope=scope,
        seq_pos=1024,
        session_key="sess-a",
        slot_id=2,
    )
    match = pool.lookup_longest_prefix(tokens, scope=scope, seq_pos=1024)
    assert match.matched_tokens == 1024
    assert match.matched_blocks == 2


def test_lookup_detects_prompt_drift(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_PREFIX_BLOCK_POOL", "1")
    scope = build_model_scope(model_hash="model1")
    pool = get_prefix_block_pool(model_scope=scope)
    tokens = _tokens(1024)
    pool.register_prefix(
        tokens,
        scope=scope,
        seq_pos=1024,
        session_key="sess-a",
        slot_id=2,
    )
    drifted = list(tokens)
    drifted[600] = 99999
    match = pool.lookup_longest_prefix(drifted, scope=scope, seq_pos=1024)
    assert match.matched_blocks == 1
    assert match.matched_tokens == 512


def test_prefix_cache_decision_blocks_hash_mismatch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_PREFIX_BLOCK_POOL", "1")
    monkeypatch.delenv("ZEROLLAMA_LLAMA_CACHE", raising=False)
    policy = resolve_prefix_cache_policy(spec_method="none")
    scope = build_model_scope(model_hash="mh")
    pool = get_prefix_block_pool(model_scope=scope)
    tokens = _tokens(1024)
    pool.register_prefix(
        tokens,
        scope=scope,
        seq_pos=1024,
        session_key="k",
        slot_id=0,
    )
    drifted = list(tokens)
    drifted[700] = 42
    allow, resume, reason = prefix_cache_decision(
        "k",
        policy,
        seq_pos=1024,
        prompt_tokens=1024,
        prompt_token_ids=drifted,
        model_hash="mh",
    )
    assert allow is False
    assert resume is None
    assert reason == "prefix_block_hash_mismatch"


def test_lmcache_tier_persists_and_hydrates(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_LMCACHE_URI", f"file://{tmp_path}")
    monkeypatch.setenv("ZEROLLAMA_PREFIX_BLOCK_POOL", "1")
    reset_lmcache_tier_for_tests()

    scope = build_model_scope(model_hash="model1")
    pool = get_prefix_block_pool(model_scope=scope)
    tokens = _tokens(512)
    pool.register_prefix(
        tokens,
        scope=scope,
        seq_pos=512,
        session_key="sess",
        slot_id=1,
        blob_path="/tmp/slot_1_0.bin",
    )
    assert lmcache_tier().health()["record_count"] == 1

    reset_prefix_block_pools_for_tests()
    match = get_prefix_block_pool(model_scope=scope).lookup_longest_prefix(
        tokens, scope=scope, seq_pos=512
    )
    assert match.matched_blocks == 1
    assert match.lmcache_hits == 1


def test_prefix_block_pool_disabled_by_default():
    assert prefix_block_pool_enabled() is False


def test_multi_holder_ref_count(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_PREFIX_BLOCK_POOL", "1")
    scope = build_model_scope(model_hash="model1")
    pool = get_prefix_block_pool(model_scope=scope)
    tokens = _tokens(1024)
    pool.register_prefix(
        tokens, scope=scope, seq_pos=1024, session_key="a", slot_id=1
    )
    pool.register_prefix(
        tokens, scope=scope, seq_pos=1024, session_key="b", slot_id=2
    )
    h = pool.health(scope=scope)
    assert h["multi_holder_blocks"] == 2
    assert isinstance(h.get("block_hashes"), list)
    assert len(h["block_hashes"]) == 2
    assert pool.release_slot_holders(1) == 0
    assert pool.health(scope=scope)["multi_holder_blocks"] == 0
    found = pool.find_donor_slot_prefix(
        tokens, scope=scope, target_slot=5, min_matched=0
    )
    assert found is not None
    assert found[0] == 2
    assert found[1] == 1024


def test_health_block_hashes_capped(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_PREFIX_BLOCK_POOL", "1")
    monkeypatch.setenv("ZEROLLAMA_RADIX_HEALTH_HASH_CAP", "1")
    scope = build_model_scope(model_hash="cap1")
    pool = get_prefix_block_pool(model_scope=scope)
    tokens = _tokens(1024)
    pool.register_prefix(tokens, scope=scope, seq_pos=1024, session_key="s", slot_id=1)
    h = pool.health(scope=scope)
    assert h["entry_count"] == 2
    assert len(h["block_hashes"]) == 1
    from runtime.kv.prefix_block_pool import prefix_block_pool_health

    agg = prefix_block_pool_health()
    assert len(agg["block_hashes"]) == 1


def test_best_donor_picks_longest_chain(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_PREFIX_BLOCK_POOL", "1")
    monkeypatch.setenv("ZEROLLAMA_RADIX_PREFIX_SHARE", "1")
    scope = build_model_scope(model_hash="m1")
    pool = get_prefix_block_pool(model_scope=scope)
    tokens = _tokens(1024)
    pool.register_prefix(
        tokens, scope=scope, seq_pos=512, session_key="short", slot_id=1
    )
    pool.register_prefix(
        tokens, scope=scope, seq_pos=1024, session_key="full", slot_id=2
    )
    from runtime.kv.radix_prefix_share import find_radix_share_plan

    plan = find_radix_share_plan(tokens, target_slot=5, model_hash="m1", seq_pos=0)
    assert plan is not None
    assert plan.source_slot == 2
    assert plan.copy_tokens == 1024

