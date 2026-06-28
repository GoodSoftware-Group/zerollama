"""Hash-chained prefix block pool for L3 cache_prompt (vLLM BlockPool-inspired).

Tracks content-addressed prefix blocks per model scope. Used to:
- verify stored KV still matches the incoming prompt token prefix;
- optionally persist block metadata to the LMCache tier;
- expose operator stats on shared prefix blocks across sessions.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from runtime.env import lmcache_tier_enabled, prefix_block_pool_enabled, prefix_block_pool_max_entries
from runtime.kv.lmcache_tier import LMCacheBlockRecord, lmcache_tier
from runtime.kv.prefix_block_hash import iter_prefix_blocks, model_scope_key
from runtime.kv_cache_spec import prefix_cache_block_size


@dataclass
class PrefixBlockEntry:
    block_hash: str
    parent_hash: str
    block_index: int
    token_end: int
    model_scope: str
    session_key: str | None
    slot_id: int | None
    tier: str = "ram"
    ref_count: int = 1
    updated_at_ms: float = field(default_factory=lambda: time.time() * 1000)
    blob_path: str | None = None


@dataclass(frozen=True)
class PrefixBlockMatch:
    matched_tokens: int
    matched_blocks: int
    tail_hash: str | None
    verified: bool
    lmcache_hits: int = 0
    donor_slot: int | None = None


class PrefixBlockPool:
    def __init__(self, *, max_entries: int) -> None:
        self.max_entries = max_entries
        self._blocks: dict[str, PrefixBlockEntry] = {}
        self._lock = threading.RLock()

    def lookup_longest_prefix(
        self,
        tokens: list[int],
        *,
        scope: str,
        seq_pos: int | None,
        block_size: int | None = None,
    ) -> PrefixBlockMatch:
        bs = max(1, int(block_size or prefix_cache_block_size()))
        limit = len(tokens)
        if seq_pos is not None and seq_pos >= 0:
            limit = min(limit, seq_pos)
        if limit <= 0 or not tokens:
            return PrefixBlockMatch(0, 0, None, verified=True)

        tier = lmcache_tier()
        matched = 0
        blocks = 0
        tail: str | None = None
        lmcache_hits = 0

        with self._lock:
            for _idx, start, end, _parent, bh in iter_prefix_blocks(
                tokens, block_size=bs, scope=scope, max_tokens=limit
            ):
                entry = self._blocks.get(bh)
                if entry is None:
                    rec = tier.get(model_scope=scope, block_hash=bh)
                    if rec is not None:
                        entry = self._hydrate_from_lmcache(rec)
                        lmcache_hits += 1
                if entry is None or entry.model_scope != scope:
                    break
                matched = end
                blocks += 1
                tail = bh
                entry.ref_count += 1
                entry.updated_at_ms = time.time() * 1000

        verified = matched >= limit or (limit % bs == 0 and matched == limit)
        if limit > matched and limit - matched < bs:
            # Tail shorter than one block — treat as verified when all full blocks match.
            verified = matched > 0 and matched == (limit // bs) * bs
        return PrefixBlockMatch(
            matched_tokens=matched,
            matched_blocks=blocks,
            tail_hash=tail,
            verified=verified or matched == 0,
            lmcache_hits=lmcache_hits,
        )

    def find_donor_slot_prefix(
        self,
        tokens: list[int],
        *,
        scope: str,
        target_slot: int,
        max_tokens: int | None = None,
        block_size: int | None = None,
    ) -> tuple[int, int, int] | None:
        """Return ``(donor_slot, matched_tokens, matched_blocks)`` for cross-slot seed.

        WHY not copy from disk alone: L3 already persists slot blobs per key; Radix
        copies live RAM KV from a warm donor when keys differ but token hash chain matches.
        """
        bs = max(1, int(block_size or prefix_cache_block_size()))
        limit = len(tokens)
        if max_tokens is not None and max_tokens >= 0:
            limit = min(limit, max_tokens)
        if limit <= 0 or not tokens or target_slot < 0:
            return None

        tier = lmcache_tier()
        matched = 0
        blocks = 0
        donor_slot: int | None = None

        with self._lock:
            for _idx, _start, end, _parent, bh in iter_prefix_blocks(
                tokens, block_size=bs, scope=scope, max_tokens=limit
            ):
                entry = self._blocks.get(bh)
                if entry is None:
                    rec = tier.get(model_scope=scope, block_hash=bh)
                    if rec is not None:
                        entry = self._hydrate_from_lmcache(rec)
                if entry is None or entry.slot_id is None:
                    break
                donor = int(entry.slot_id)
                if donor == target_slot:
                    break
                if donor_slot is None:
                    donor_slot = donor
                elif donor != donor_slot:
                    break
                matched = end
                blocks += 1

        if donor_slot is None or matched <= 0 or blocks <= 0:
            return None
        return donor_slot, matched, blocks

    def register_prefix(
        self,
        tokens: list[int],
        *,
        scope: str,
        seq_pos: int,
        session_key: str | None,
        slot_id: int | None,
        blob_path: str | None = None,
        block_size: int | None = None,
    ) -> list[str]:
        if seq_pos <= 0 or not tokens:
            return []
        bs = max(1, int(block_size or prefix_cache_block_size()))
        tier = lmcache_tier()
        now = time.time() * 1000
        out: list[str] = []

        with self._lock:
            for _idx, _start, end, parent, bh in iter_prefix_blocks(
                tokens, block_size=bs, scope=scope, max_tokens=seq_pos
            ):
                entry = PrefixBlockEntry(
                    block_hash=bh,
                    parent_hash=parent,
                    block_index=_idx,
                    token_end=end,
                    model_scope=scope,
                    session_key=session_key,
                    slot_id=slot_id,
                    tier="lmcache" if lmcache_tier_enabled() else "ram",
                    updated_at_ms=now,
                    blob_path=blob_path,
                )
                self._blocks[bh] = entry
                out.append(bh)
                if lmcache_tier_enabled():
                    tier.put(
                        LMCacheBlockRecord(
                            block_hash=bh,
                            parent_hash=parent,
                            block_index=_idx,
                            token_end=end,
                            model_scope=scope,
                            session_key=session_key,
                            slot_id=slot_id,
                            blob_path=blob_path,
                            updated_at_ms=int(now),
                        )
                    )
            self._evict_if_needed()

        return out

    def _hydrate_from_lmcache(self, rec: LMCacheBlockRecord) -> PrefixBlockEntry:
        entry = PrefixBlockEntry(
            block_hash=rec.block_hash,
            parent_hash=rec.parent_hash,
            block_index=rec.block_index,
            token_end=rec.token_end,
            model_scope=rec.model_scope,
            session_key=rec.session_key,
            slot_id=rec.slot_id,
            tier="lmcache",
            blob_path=rec.blob_path,
            updated_at_ms=float(rec.updated_at_ms or time.time() * 1000),
        )
        self._blocks[rec.block_hash] = entry
        return entry

    def _evict_if_needed(self) -> None:
        overflow = len(self._blocks) - self.max_entries
        if overflow <= 0:
            return
        victims = sorted(
            self._blocks.values(),
            key=lambda e: (e.ref_count, e.updated_at_ms),
        )[:overflow]
        for entry in victims:
            self._blocks.pop(entry.block_hash, None)

    def health(self, *, scope: str | None = None) -> dict[str, Any]:
        with self._lock:
            entries = list(self._blocks.values())
        if scope is not None:
            entries = [e for e in entries if e.model_scope == scope]
        sessions = {e.session_key for e in entries if e.session_key}
        slots = {e.slot_id for e in entries if e.slot_id is not None}
        return {
            "enabled": prefix_block_pool_enabled(),
            "block_size": prefix_cache_block_size(),
            "max_entries": self.max_entries,
            "entry_count": len(entries),
            "session_count": len(sessions),
            "slot_count": len(slots),
            "lmcache_tier": lmcache_tier().health(),
        }


_POOLS_LOCK = threading.Lock()
_POOLS: dict[str, PrefixBlockPool] = {}


def get_prefix_block_pool(*, model_scope: str) -> PrefixBlockPool:
    with _POOLS_LOCK:
        pool = _POOLS.get(model_scope)
        if pool is None:
            pool = PrefixBlockPool(max_entries=prefix_block_pool_max_entries())
            _POOLS[model_scope] = pool
        return pool


def prefix_block_pool_health(*, model_scope: str | None = None) -> dict[str, Any]:
    if not prefix_block_pool_enabled():
        return {
            "enabled": False,
            "block_size": prefix_cache_block_size(),
            "lmcache_tier": lmcache_tier().health(),
        }
    if model_scope:
        return get_prefix_block_pool(model_scope=model_scope).health(scope=model_scope)
    with _POOLS_LOCK:
        scopes = list(_POOLS.keys())
    total = 0
    for scope in scopes:
        total += get_prefix_block_pool(model_scope=scope).health(scope=scope)[
            "entry_count"
        ]
    return {
        "enabled": True,
        "block_size": prefix_cache_block_size(),
        "scope_count": len(scopes),
        "entry_count": total,
        "lmcache_tier": lmcache_tier().health(),
    }


def reset_prefix_block_pools_for_tests() -> None:
    with _POOLS_LOCK:
        _POOLS.clear()


def build_model_scope(
    *,
    model_hash: str,
    cache_salt: str | None = None,
) -> str:
    return model_scope_key(model_hash=model_hash, cache_salt=cache_salt)
