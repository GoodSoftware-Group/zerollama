"""Tests for cache_level / cache_reset helpers and Radix prefer-parent donor."""

from __future__ import annotations

from runtime.cache_bridge import (
    apply_cache_level_to_policy,
    extract_cache_level,
    extract_cache_reset,
    extract_session_group,
    extract_session_parent,
)
from runtime.kv.prefix_block_pool import PrefixBlockPool, _PrefixChainBlock
from runtime.prefix_cache_policy import PrefixCachePolicy


def test_extract_zerollama_cache_fields():
    opts = {
        "zerollama": {
            "session_parent": "parent:1",
            "session_group": "hermes",
            "cache_reset": True,
            "cache_level": "vram",
        }
    }
    assert extract_session_parent(opts) == "parent:1"
    assert extract_session_group(opts) == "hermes"
    assert extract_cache_reset(opts) is True
    assert extract_cache_level(opts) == "gpu"


def test_apply_cache_level_to_policy():
    base = PrefixCachePolicy(
        kind="standard",
        allow_cache_prompt=True,
        allow_disk_persist=True,
        effective_window=None,
        disk_ttl_ms=3600000,
        speculative_draft=False,
    )
    assert apply_cache_level_to_policy(base, "auto").allow_disk_persist is True
    assert apply_cache_level_to_policy(base, "dram").allow_disk_persist is False
    assert apply_cache_level_to_policy(base, "disk").allow_disk_persist is True

    draft = PrefixCachePolicy(
        kind="standard",
        allow_cache_prompt=True,
        allow_disk_persist=False,
        effective_window=None,
        disk_ttl_ms=3600000,
        speculative_draft=True,
    )
    # Hard deny: disk request must not override draft-spec.
    assert apply_cache_level_to_policy(draft, "disk").allow_disk_persist is False


def test_best_donor_prefers_parent_on_tie():
    # Two candidates with equal matched length; prefer parent's slot.
    chain = [
        _PrefixChainBlock(token_end=128, holders=frozenset({1, 2})),
        _PrefixChainBlock(token_end=256, holders=frozenset({1, 2})),
    ]
    meta = {
        1: ("other:key", "g"),
        2: ("parent:key", "g"),
    }
    found = PrefixBlockPool._best_donor_from_chain(
        chain,
        target_slot=0,
        skip_slot=0,
        min_matched=0,
        prefer_session_key="parent:key",
        prefer_session_group="g",
        slot_meta=meta,
    )
    assert found is not None
    assert found[0] == 2
    assert found[1] == 256


def test_best_donor_prefers_group_when_parent_absent():
    chain = [
        _PrefixChainBlock(token_end=64, holders=frozenset({3, 4})),
    ]
    meta = {
        3: ("a", "other"),
        4: ("b", "wanted"),
    }
    found = PrefixBlockPool._best_donor_from_chain(
        chain,
        target_slot=0,
        skip_slot=0,
        min_matched=0,
        prefer_session_key="missing",
        prefer_session_group="wanted",
        slot_meta=meta,
    )
    assert found is not None
    assert found[0] == 4
