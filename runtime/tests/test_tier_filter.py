"""Per-request load-tier filter (vLLM #48123)."""

from __future__ import annotations

import pytest

from runtime.env import reset_runtime_env_for_tests
from runtime.kv.lmcache_tier import (
    LMCacheBlockRecord,
    reset_lmcache_tier_for_tests,
)
from runtime.kv.prefix_block_pool import (
    build_model_scope,
    get_prefix_block_pool,
    reset_prefix_block_pools_for_tests,
)
from runtime.kv.tier_filter import (
    Locality,
    Medium,
    TierFilter,
    TierMatcher,
    parse_tier_filter,
)


@pytest.fixture(autouse=True)
def _reset(tmp_path, monkeypatch: pytest.MonkeyPatch):
    reset_runtime_env_for_tests()
    reset_prefix_block_pools_for_tests()
    reset_lmcache_tier_for_tests()
    monkeypatch.setenv("ZEROLLAMA_PREFIX_BLOCK_POOL", "1")
    monkeypatch.setenv("ZEROLLAMA_LMCACHE_URI", f"file://{tmp_path / 'lmcache'}")
    yield
    reset_runtime_env_for_tests()
    reset_prefix_block_pools_for_tests()
    reset_lmcache_tier_for_tests()


def test_parse_tier_filter_all_empty_deny():
    assert parse_tier_filter(None) is TierFilter.ALL
    deny = parse_tier_filter([])
    assert deny is not TierFilter.ALL
    assert not deny.allows_lmcache()
    assert not deny.allows(Medium.CPU, Locality.LOCAL)


def test_parse_tier_filter_storage_only():
    f = parse_tier_filter([{"medium": "STORAGE"}])
    assert f.allows_lmcache()
    assert not f.allows(Medium.CPU, Locality.LOCAL)


def test_lookup_skips_lmcache_when_filter_denies(monkeypatch: pytest.MonkeyPatch):
    scope = build_model_scope(model_hash="tier-a")
    pool = get_prefix_block_pool(model_scope=scope)
    tokens = list(range(512))
    # Seed only via LMCache tier (no RAM holders → hydrate path).
    from runtime.kv.lmcache_tier import lmcache_tier
    from runtime.kv.prefix_block_hash import iter_prefix_blocks

    blocks = list(iter_prefix_blocks(tokens, block_size=512, scope=scope))
    assert len(blocks) == 1
    _idx, _s, end, parent, bh = blocks[0]
    lmcache_tier().put(
        LMCacheBlockRecord(
            block_hash=bh,
            parent_hash=parent,
            block_index=_idx,
            token_end=end,
            model_scope=scope,
            session_key="s",
            slot_id=3,
            updated_at_ms=1,
        )
    )
    # Without filter: hydrate works but holders empty → chain breaks.
    # Register properly then clear RAM to force hydrate.
    pool.register_prefix(
        tokens, scope=scope, seq_pos=512, session_key="s", slot_id=3, block_size=512
    )
    pool._blocks.clear()

    hit = pool.lookup_longest_prefix(tokens, scope=scope, seq_pos=512)
    assert hit.lmcache_hits >= 1
    assert hit.matched_tokens == 512

    pool._blocks.clear()
    deny = TierFilter(matchers=())  # deny all secondaries
    miss = pool.lookup_longest_prefix(
        tokens, scope=scope, seq_pos=512, load_tier_filter=deny
    )
    assert miss.lmcache_hits == 0
    assert miss.matched_tokens == 0


def test_tier_matcher_wildcard():
    m = TierMatcher()
    assert m.matches(Medium.STORAGE, Locality.REMOTE)
    assert TierMatcher(medium=Medium.CPU).matches(Medium.CPU, Locality.LOCAL)
    assert not TierMatcher(medium=Medium.CPU).matches(Medium.STORAGE, Locality.LOCAL)
