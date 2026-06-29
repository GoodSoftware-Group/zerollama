"""LMCache tier backends (file + redis)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from runtime.kv.lmcache_redis import RedisConfig, RedisLMCacheTierStore, parse_redis_uri
from runtime.kv.lmcache_tier import (
    LMCacheBlockRecord,
    _safe_dir_name,
    lmcache_tier,
    reset_lmcache_tier_for_tests,
)
from runtime.kv.prefix_block_hash import iter_prefix_blocks
from runtime.kv.prefix_block_pool import (
    build_model_scope,
    get_prefix_block_pool,
    reset_prefix_block_pools_for_tests,
)


def _tokens(n: int) -> list[int]:
    return [100 + i for i in range(n)]


def test_parse_redis_uri():
    cfg = parse_redis_uri("redis://:secret@cache.local:6380/2")
    assert cfg.host == "cache.local"
    assert cfg.port == 6380
    assert cfg.db == 2
    assert cfg.password == "secret"


def test_redis_tier_put_get_roundtrip():
    storage: dict[str, str] = {}
    mock_client = MagicMock()
    mock_client.get.side_effect = lambda k: storage.get(k)
    mock_client.set.side_effect = lambda k, v, ttl_sec=None: storage.__setitem__(k, v)

    cfg = RedisConfig(
        host="127.0.0.1",
        port=6379,
        db=0,
        password=None,
        key_prefix="zerollama:lmcache:v1",
        ttl_sec=3600,
    )
    store = RedisLMCacheTierStore(cfg, uri="redis://127.0.0.1:6379/0")
    store._client = mock_client  # type: ignore[method-assign]

    rec = LMCacheBlockRecord(
        block_hash="abc" * 21 + "a",
        parent_hash="0" * 64,
        block_index=0,
        token_end=512,
        model_scope="scope1",
        session_key="s",
        slot_id=1,
    )
    store.put(rec)
    got = store.get(model_scope="scope1", block_hash=rec.block_hash)
    assert got is not None
    assert got.slot_id == 1
    mock_client.set.assert_called_once()


@pytest.fixture(autouse=True)
def _reset():
    reset_lmcache_tier_for_tests()
    reset_prefix_block_pools_for_tests()
    yield
    reset_lmcache_tier_for_tests()
    reset_prefix_block_pools_for_tests()


def test_lmcache_tier_factory_selects_redis(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_LMCACHE_URI", "redis://127.0.0.1:6379/0")
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_client.dbsize.return_value = 0
    cfg = parse_redis_uri("redis://127.0.0.1:6379/0")
    tier = RedisLMCacheTierStore(cfg, uri="redis://127.0.0.1:6379/0")
    tier._client = mock_client  # type: ignore[method-assign]
    with patch("runtime.kv.lmcache_tier._create_tier_store", return_value=tier):
        reset_lmcache_tier_for_tests()
        h = lmcache_tier().health()
        assert h["backend"] == "redis"
        assert h["reachable"] is True


def test_prefix_pool_hydrates_from_redis_tier(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_LMCACHE_URI", "redis://127.0.0.1:6379/0")
    monkeypatch.setenv("ZEROLLAMA_PREFIX_BLOCK_POOL", "1")

    scope = build_model_scope(model_hash="mh")
    tokens = _tokens(512)
    blocks = list(iter_prefix_blocks(tokens, block_size=512, scope=scope, max_tokens=512))
    assert len(blocks) == 1
    bh = blocks[0][4]

    storage: dict[str, str] = {}
    rec = LMCacheBlockRecord(
        block_hash=bh,
        parent_hash=blocks[0][3],
        block_index=0,
        token_end=512,
        model_scope=scope,
        session_key="remote",
        slot_id=7,
    )
    storage[f"zerollama:lmcache:v1:{_safe_dir_name(scope)}:{bh}"] = json.dumps(
        rec.to_dict()
    )

    mock_client = MagicMock()
    mock_client.get.side_effect = lambda k: storage.get(k)
    mock_client.ping.return_value = True
    mock_client.dbsize.return_value = len(storage)

    cfg = parse_redis_uri("redis://127.0.0.1:6379/0")
    tier = RedisLMCacheTierStore(cfg, uri="redis://127.0.0.1:6379/0")
    tier._client = mock_client  # type: ignore[method-assign]

    with patch("runtime.kv.lmcache_tier._create_tier_store", return_value=tier):
        reset_lmcache_tier_for_tests()
        match = get_prefix_block_pool(model_scope=scope).lookup_longest_prefix(
            tokens, scope=scope, seq_pos=512
        )
        assert match.matched_tokens == 512
        assert match.lmcache_hits == 1
