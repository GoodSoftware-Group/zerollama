"""Hash-chained prefix block pool for L3 cache_prompt (vLLM BlockPool-inspired).

WHY this pool exists:
  - Same-key L3: verify stored KV still matches incoming tokens before ``cache_prompt``.
  - Radix: find a *donor slot* whose registered blocks match the target's prefix
    hash chain so we can ``seq_cp`` live RAM KV (disk blobs alone are per-key).

WHY multi-holder ref_count (L3-R3): identical prefix blocks may be registered by
several slots; eviction must not drop metadata while any slot still references it.
Donor search picks the slot with the longest contiguous chain from token 0.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
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
    holder_slots: frozenset[int] = field(default_factory=frozenset)
    updated_at_ms: float = field(default_factory=lambda: time.time() * 1000)
    blob_path: str | None = None


def _holder_slots(entry: PrefixBlockEntry) -> frozenset[int]:
    if entry.holder_slots:
        return entry.holder_slots
    if entry.slot_id is not None:
        return frozenset({int(entry.slot_id)})
    return frozenset()


@dataclass(frozen=True)
class PrefixBlockMatch:
    matched_tokens: int
    matched_blocks: int
    tail_hash: str | None
    verified: bool
    lmcache_hits: int = 0
    donor_slot: int | None = None


@dataclass(frozen=True)
class _PrefixChainBlock:
    token_end: int
    holders: frozenset[int]


class PrefixBlockPool:
    def __init__(self, *, max_entries: int) -> None:
        self.max_entries = max_entries
        self._blocks: dict[str, PrefixBlockEntry] = {}
        self._lock = threading.RLock()

    def _resolve_entry(
        self,
        bh: str,
        *,
        scope: str,
        tier: Any,
    ) -> PrefixBlockEntry | None:
        entry = self._blocks.get(bh)
        if entry is None:
            rec = tier.get(model_scope=scope, block_hash=bh)
            if rec is not None:
                entry = self._hydrate_from_lmcache(rec)
        if entry is None or entry.model_scope != scope:
            return None
        return entry

    def _build_prefix_chain(
        self,
        tokens: list[int],
        *,
        scope: str,
        limit: int,
        block_size: int | None = None,
    ) -> tuple[list[_PrefixChainBlock], int]:
        bs = max(1, int(block_size or prefix_cache_block_size()))
        tier = lmcache_tier()
        chain: list[_PrefixChainBlock] = []
        lmcache_hits = 0
        with self._lock:
            for _idx, _start, end, _parent, bh in iter_prefix_blocks(
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
                holders = _holder_slots(entry)
                if not holders:
                    break
                chain.append(_PrefixChainBlock(token_end=end, holders=holders))
        return chain, lmcache_hits

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

        chain, lmcache_hits = self._build_prefix_chain(
            tokens, scope=scope, limit=limit, block_size=bs
        )
        matched = chain[-1].token_end if chain else 0
        blocks = len(chain)
        tail: str | None = None
        if chain:
            bs = max(1, int(block_size or prefix_cache_block_size()))
            with self._lock:
                idx = 0
                for _idx, _start, end, _parent, bh in iter_prefix_blocks(
                    tokens, block_size=bs, scope=scope, max_tokens=limit
                ):
                    if idx >= len(chain):
                        break
                    entry = self._blocks.get(bh)
                    if entry is not None:
                        tail = bh
                        entry.ref_count += 1
                        entry.updated_at_ms = time.time() * 1000
                    idx += 1

        verified = matched >= limit or (limit % bs == 0 and matched == limit)
        if limit > matched and limit - matched < bs:
            verified = matched > 0 and matched == (limit // bs) * bs
        return PrefixBlockMatch(
            matched_tokens=matched,
            matched_blocks=blocks,
            tail_hash=tail,
            verified=verified or matched == 0,
            lmcache_hits=lmcache_hits,
        )

    def verify_target_slot_prefix(
        self,
        tokens: list[int],
        *,
        scope: str,
        target_slot: int,
        seq_pos: int,
        block_size: int | None = None,
    ) -> bool:
        """True when block metadata for ``tokens[:seq_pos]`` includes ``target_slot``.

        WHY (L3-R2): warm catch-up must prove the target slot *owns* the prefix
        blocks, not merely that the hash exists on another slot — otherwise we
        copy donor KV over an unverified partial cache.
        """
        if seq_pos <= 0 or target_slot < 0 or not tokens:
            return seq_pos == 0
        bs = max(1, int(block_size or prefix_cache_block_size()))
        limit = min(len(tokens), seq_pos)
        if limit <= 0:
            return False
        chain, _ = self._build_prefix_chain(
            tokens, scope=scope, limit=limit, block_size=bs
        )
        if not chain:
            return False
        matched = chain[-1].token_end
        for block in chain:
            if target_slot not in block.holders:
                return False
        return matched >= limit or (limit % bs == 0 and matched >= (limit // bs) * bs)

    @staticmethod
    def _best_donor_from_chain(
        chain: list[_PrefixChainBlock],
        *,
        target_slot: int,
        skip_slot: int | None,
        min_matched: int,
    ) -> tuple[int, int, int] | None:
        """Pick donor with longest contiguous prefix from token 0 (L3-R3).

        ``skip_slot`` (usually target): blocks held only by skip still advance the
        chain position for warm catch-up; donor must hold every non-skip block.
        """
        if not chain:
            return None
        skip = skip_slot if skip_slot is not None else target_slot
        candidates: set[int] = set()
        for block in chain:
            candidates.update(block.holders)
        candidates.discard(target_slot)
        if skip != target_slot:
            candidates.discard(skip)

        best: tuple[int, int, int] | None = None
        for cand in sorted(candidates):
            matched = 0
            blocks = 0
            for block in chain:
                if cand in block.holders:
                    matched = block.token_end
                    blocks += 1
                    continue
                if skip in block.holders:
                    matched = block.token_end
                    blocks += 1
                    continue
                break
            if matched <= min_matched:
                continue
            if best is None or matched > best[1]:
                best = (cand, matched, blocks)
        return best

    def find_donor_slot_prefix(
        self,
        tokens: list[int],
        *,
        scope: str,
        target_slot: int,
        max_tokens: int | None = None,
        block_size: int | None = None,
        exclude_slot: int | None = None,
        min_matched: int = 0,
    ) -> tuple[int, int, int] | None:
        """Return ``(donor_slot, matched_tokens, matched_blocks)`` for cross-slot seed.

        ``exclude_slot`` / ``min_matched`` (L3-R2): skip target-owned blocks while
        walking the hash chain so a warm target can catch up from a longer donor.
        """
        limit = len(tokens)
        if max_tokens is not None and max_tokens >= 0:
            limit = min(limit, max_tokens)
        if limit <= 0 or not tokens or target_slot < 0:
            return None
        chain, _ = self._build_prefix_chain(
            tokens,
            scope=scope,
            limit=limit,
            block_size=block_size,
        )
        return self._best_donor_from_chain(
            chain,
            target_slot=target_slot,
            skip_slot=exclude_slot,
            min_matched=min_matched,
        )

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
        if seq_pos <= 0 or not tokens or slot_id is None:
            return []
        bs = max(1, int(block_size or prefix_cache_block_size()))
        tier = lmcache_tier()
        now = time.time() * 1000
        sid = int(slot_id)
        out: list[str] = []

        with self._lock:
            for _idx, _start, end, parent, bh in iter_prefix_blocks(
                tokens, block_size=bs, scope=scope, max_tokens=seq_pos
            ):
                existing = self._blocks.get(bh)
                holders = frozenset({sid})
                if existing is not None:
                    holders = frozenset(set(_holder_slots(existing)) | {sid})
                entry = PrefixBlockEntry(
                    block_hash=bh,
                    parent_hash=parent,
                    block_index=_idx,
                    token_end=end,
                    model_scope=scope,
                    session_key=session_key,
                    slot_id=sid,
                    tier="lmcache" if lmcache_tier_enabled() else "ram",
                    holder_slots=holders,
                    ref_count=len(holders),
                    updated_at_ms=now,
                    blob_path=blob_path or (existing.blob_path if existing else None),
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
                            slot_id=sid,
                            blob_path=entry.blob_path,
                            updated_at_ms=int(now),
                        )
                    )
            self._evict_if_needed()

        return out

    def release_slot_holders(self, slot_id: int) -> int:
        """Drop ``slot_id`` from all block holder sets; remove entries with no holders.

        WHY (L3-R3): slot teardown must not leave ghost holders or evict blocks
        still referenced by another slot's Radix chain.
        """
        if slot_id < 0:
            return 0
        sid = int(slot_id)
        removed = 0
        with self._lock:
            for bh, entry in list(self._blocks.items()):
                holders = set(_holder_slots(entry))
                if sid not in holders:
                    continue
                holders.discard(sid)
                if not holders:
                    self._blocks.pop(bh, None)
                    removed += 1
                else:
                    self._blocks[bh] = replace(
                        entry,
                        holder_slots=frozenset(holders),
                        ref_count=len(holders),
                        slot_id=next(iter(holders)) if entry.slot_id == sid else entry.slot_id,
                    )
        return removed

    def _hydrate_from_lmcache(self, rec: LMCacheBlockRecord) -> PrefixBlockEntry:
        sid = int(rec.slot_id) if rec.slot_id is not None else None
        holders = frozenset({sid}) if sid is not None else frozenset()
        entry = PrefixBlockEntry(
            block_hash=rec.block_hash,
            parent_hash=rec.parent_hash,
            block_index=rec.block_index,
            token_end=rec.token_end,
            model_scope=rec.model_scope,
            session_key=rec.session_key,
            slot_id=sid,
            tier="lmcache",
            holder_slots=holders,
            ref_count=len(holders) or 1,
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
            key=lambda e: (len(_holder_slots(e)), e.ref_count, e.updated_at_ms),
        )
        for entry in victims:
            if overflow <= 0:
                break
            if len(_holder_slots(entry)) > 0:
                continue
            self._blocks.pop(entry.block_hash, None)
            overflow -= 1
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
        slots: set[int] = set()
        for e in entries:
            slots.update(_holder_slots(e))
        multi_holder = sum(1 for e in entries if len(_holder_slots(e)) > 1)
        return {
            "enabled": prefix_block_pool_enabled(),
            "block_size": prefix_cache_block_size(),
            "max_entries": self.max_entries,
            "entry_count": len(entries),
            "session_count": len(sessions),
            "slot_count": len(slots),
            "multi_holder_blocks": multi_holder,
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
    multi = 0
    for scope in scopes:
        h = get_prefix_block_pool(model_scope=scope).health(scope=scope)
        total += h["entry_count"]
        multi += h.get("multi_holder_blocks", 0)
    return {
        "enabled": True,
        "block_size": prefix_cache_block_size(),
        "scope_count": len(scopes),
        "entry_count": total,
        "multi_holder_blocks": multi,
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
