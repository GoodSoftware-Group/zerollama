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

import os
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from runtime.env import lmcache_tier_enabled, prefix_block_pool_enabled, prefix_block_pool_max_entries
from runtime.kv.lmcache_tier import LMCacheBlockRecord, lmcache_tier
from runtime.kv.prefix_block_hash import iter_prefix_blocks, model_scope_key
from runtime.kv.tier_filter import TierFilter, lmcache_is_remote
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
    session_group: str | None = None
    tier: str = "ram"
    ref_count: int = 1
    holder_slots: frozenset[int] = field(default_factory=frozenset)
    updated_at_ms: float = field(default_factory=lambda: time.time() * 1000)
    blob_path: str | None = None
    blob_digest: str | None = None


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
    # True when secondary tier returned a prefix shorter than requested (vLLM #50321).
    partial_tier_load: bool = False


@dataclass(frozen=True)
class _PrefixChainBlock:
    token_end: int
    holders: frozenset[int]
    tier_lmcache: bool = False


@dataclass(frozen=True)
class PrefixBlobMatch:
    """Longest prefix with a federated blob digest (L3-R7 cold restore)."""

    matched_tokens: int
    matched_blocks: int
    tail_hash: str | None
    blob_digest: str
    source_slot_id: int | None = None


@dataclass(frozen=True)
class RegisterPrefixResult:
    """Outcome of ``register_prefix`` (vLLM finish-time / creation accounting)."""

    block_hashes: list[str] = field(default_factory=list)
    registered_tokens: int = 0
    blob_digest: str | None = None
    blob_finalized: bool = False
    skipped_swa_blocks: int = 0

    def __iter__(self):
        return iter(self.block_hashes)

    def __len__(self) -> int:
        return len(self.block_hashes)

    def __bool__(self) -> bool:
        return bool(self.block_hashes)

    def __getitem__(self, index: int) -> str:
        return self.block_hashes[index]


# Pending blob finalize after metadata register (vLLM #48596/#49671).
# Keyed by (model_scope, slot_id) → blob_path awaiting publish.
_pending_blob_finalize: dict[tuple[str, int], str] = {}
_pending_lock = threading.Lock()


def pending_blob_finalize_count() -> int:
    with _pending_lock:
        return len(_pending_blob_finalize)


def reset_pending_blob_finalize_for_tests() -> None:
    with _pending_lock:
        _pending_blob_finalize.clear()


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
        load_tier_filter: TierFilter | None = None,
    ) -> PrefixBlockEntry | None:
        entry = self._blocks.get(bh)
        filt = load_tier_filter or TierFilter.ALL
        if entry is None and filt.allows_lmcache(remote=lmcache_is_remote()):
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
        load_tier_filter: TierFilter | None = None,
    ) -> tuple[list[_PrefixChainBlock], int]:
        bs = max(1, int(block_size or prefix_cache_block_size()))
        tier = lmcache_tier()
        filt = load_tier_filter or TierFilter.ALL
        allow_lmcache = filt.allows_lmcache(remote=lmcache_is_remote())
        chain: list[_PrefixChainBlock] = []
        lmcache_hits = 0
        with self._lock:
            for _idx, _start, end, _parent, bh in iter_prefix_blocks(
                tokens, block_size=bs, scope=scope, max_tokens=limit
            ):
                entry = self._blocks.get(bh)
                if entry is None and allow_lmcache:
                    rec = tier.get(model_scope=scope, block_hash=bh)
                    if rec is not None:
                        entry = self._hydrate_from_lmcache(rec)
                        lmcache_hits += 1
                if entry is None or entry.model_scope != scope:
                    break
                holders = _holder_slots(entry)
                if not holders:
                    break
                chain.append(
                    _PrefixChainBlock(
                        token_end=end,
                        holders=holders,
                        tier_lmcache=(entry.tier == "lmcache"),
                    )
                )
        return chain, lmcache_hits

    def lookup_longest_prefix(
        self,
        tokens: list[int],
        *,
        scope: str,
        seq_pos: int | None,
        block_size: int | None = None,
        load_tier_filter: TierFilter | None = None,
    ) -> PrefixBlockMatch:
        bs = max(1, int(block_size or prefix_cache_block_size()))
        limit = len(tokens)
        if seq_pos is not None and seq_pos >= 0:
            limit = min(limit, seq_pos)
        if limit <= 0 or not tokens:
            return PrefixBlockMatch(0, 0, None, verified=True)

        chain, lmcache_hits = self._build_prefix_chain(
            tokens,
            scope=scope,
            limit=limit,
            block_size=bs,
            load_tier_filter=load_tier_filter,
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
        partial_tier = (
            matched > 0
            and matched < limit
            and blocks < (limit // bs)
            and (lmcache_hits > 0 or any(b.tier_lmcache for b in chain))
        )
        return PrefixBlockMatch(
            matched_tokens=matched,
            matched_blocks=blocks,
            tail_hash=tail,
            verified=verified or matched == 0,
            lmcache_hits=lmcache_hits,
            partial_tier_load=partial_tier,
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
        prefer_session_key: str | None = None,
        prefer_session_group: str | None = None,
        slot_meta: dict[int, tuple[str | None, str | None]] | None = None,
    ) -> tuple[int, int, int] | None:
        """Pick donor with longest contiguous prefix from token 0 (L3-R3).

        ``skip_slot`` (usually target): blocks held only by skip still advance the
        chain position for warm catch-up; donor must hold every non-skip block.

        When matched lengths tie, prefer ``prefer_session_key`` (session_parent)
        then ``prefer_session_group`` over arbitrary slot order.
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

        prefer_key = (prefer_session_key or "").strip() or None
        prefer_group = (prefer_session_group or "").strip() or None
        meta = slot_meta or {}

        def _rank(slot: int) -> tuple[int, int]:
            # Lower is better. (0,0)=parent key, (0,1)=group, (1,0)=other.
            sk, sg = meta.get(int(slot), (None, None))
            if prefer_key and sk and sk == prefer_key:
                return (0, 0)
            if prefer_group and sg and sg == prefer_group:
                return (0, 1)
            return (1, 0)

        best: tuple[int, int, int] | None = None
        best_rank = (2, 0)
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
            rank = _rank(cand)
            if best is None or matched > best[1] or (matched == best[1] and rank < best_rank):
                best = (cand, matched, blocks)
                best_rank = rank
        return best

    def _slot_session_meta_locked(self) -> dict[int, tuple[str | None, str | None]]:
        out: dict[int, tuple[str | None, str | None]] = {}
        for entry in self._blocks.values():
            if entry.slot_id is None:
                continue
            sid = int(entry.slot_id)
            prev = out.get(sid)
            sk = entry.session_key
            sg = getattr(entry, "session_group", None)
            if prev is None:
                out[sid] = (sk, sg)
                continue
            # Prefer non-empty metadata when merging multiple blocks for a slot.
            out[sid] = (sk or prev[0], sg or prev[1])
        return out

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
        prefer_session_key: str | None = None,
        prefer_session_group: str | None = None,
        load_tier_filter: TierFilter | None = None,
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
            load_tier_filter=load_tier_filter,
        )
        with self._lock:
            slot_meta = self._slot_session_meta_locked()
        return self._best_donor_from_chain(
            chain,
            target_slot=target_slot,
            skip_slot=exclude_slot,
            min_matched=min_matched,
            prefer_session_key=prefer_session_key,
            prefer_session_group=prefer_session_group,
            slot_meta=slot_meta,
        )

    def find_blob_prefix(
        self,
        tokens: list[int],
        *,
        scope: str,
        max_tokens: int | None = None,
        min_matched: int = 0,
        block_size: int | None = None,
        load_tier_filter: TierFilter | None = None,
    ) -> PrefixBlobMatch | None:
        """Longest contiguous prefix that has a federated ``blob_digest`` (L3-R7).

        WHY separate from donor search: hydrated Redis records carry a remote
        ``slot_id`` that is meaningless locally; live ``seq_cp`` fails, but the
        content-addressed blob can still restore the target slot from disk.
        """
        from runtime.kv.lmcache_blob import lmcache_blobs_enabled

        if not lmcache_blobs_enabled() or not tokens:
            return None
        filt = load_tier_filter or TierFilter.ALL
        if not filt.allows_lmcache(remote=lmcache_is_remote()):
            return None
        limit = len(tokens) if max_tokens is None else min(len(tokens), int(max_tokens))
        if limit <= 0:
            return None
        bs = max(1, int(block_size or prefix_cache_block_size()))
        tier = lmcache_tier()
        best: PrefixBlobMatch | None = None
        with self._lock:
            for idx, _start, end, _parent, bh in iter_prefix_blocks(
                tokens, block_size=bs, scope=scope, max_tokens=limit
            ):
                entry = self._blocks.get(bh)
                if entry is None:
                    rec = tier.get(model_scope=scope, block_hash=bh)
                    if rec is not None:
                        entry = self._hydrate_from_lmcache(rec)
                if entry is None or entry.model_scope != scope:
                    break
                digest = (entry.blob_digest or "").strip()
                if not digest:
                    # Keep walking — earlier blocks may lack digest; later may have it.
                    continue
                if end < min_matched:
                    continue
                best = PrefixBlobMatch(
                    matched_tokens=end,
                    matched_blocks=idx + 1,
                    tail_hash=bh,
                    blob_digest=digest,
                    source_slot_id=entry.slot_id,
                )
        return best

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
        session_group: str | None = None,
        finalize_blob: bool | None = None,
        store_block_mask: list[bool] | None = None,
    ) -> RegisterPrefixResult:
        """Register full prefix blocks; optionally defer blob publish (vLLM #48596).

        ``finalize_blob``:
          - ``True`` — publish immediately when ``blob_path`` set
          - ``False`` — metadata only; queue pending finalize
          - ``None`` (auto) — publish only when ``blob_path`` exists on disk

        ``store_block_mask``: SWA reachable-tail filter; ``None`` stores all
        full blocks. Masked-out blocks skip pool + LMCache put.
        """
        if seq_pos <= 0 or not tokens or slot_id is None:
            return RegisterPrefixResult()
        bs = max(1, int(block_size or prefix_cache_block_size()))
        tier = lmcache_tier()
        now = time.time() * 1000
        sid = int(slot_id)
        out: list[str] = []
        skipped_swa = 0
        registered_tokens = 0
        blob_digest: str | None = None
        do_finalize = finalize_blob
        if do_finalize is None:
            do_finalize = bool(blob_path) and Path(blob_path).is_file()
        if do_finalize and blob_path and lmcache_tier_enabled():
            from runtime.kv.lmcache_blob import publish_slot_blob

            blob_digest = publish_slot_blob(blob_path)

        with self._lock:
            for _idx, _start, end, parent, bh in iter_prefix_blocks(
                tokens, block_size=bs, scope=scope, max_tokens=seq_pos
            ):
                if store_block_mask is not None:
                    if _idx >= len(store_block_mask) or not store_block_mask[_idx]:
                        skipped_swa += 1
                        continue
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
                    session_group=session_group
                    or (getattr(existing, "session_group", None) if existing else None),
                    tier="lmcache" if lmcache_tier_enabled() else "ram",
                    holder_slots=holders,
                    ref_count=len(holders),
                    updated_at_ms=now,
                    blob_path=blob_path or (existing.blob_path if existing else None),
                    blob_digest=blob_digest
                    or (existing.blob_digest if existing else None),
                )
                self._blocks[bh] = entry
                out.append(bh)
                registered_tokens = end
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
                            blob_digest=entry.blob_digest,
                            updated_at_ms=int(now),
                        )
                    )
            self._evict_if_needed()

        blob_finalized = bool(blob_digest)
        if blob_path and lmcache_tier_enabled() and not blob_finalized:
            with _pending_lock:
                _pending_blob_finalize[(scope, sid)] = str(blob_path)
        elif blob_finalized:
            with _pending_lock:
                _pending_blob_finalize.pop((scope, sid), None)

        return RegisterPrefixResult(
            block_hashes=out,
            registered_tokens=registered_tokens,
            blob_digest=blob_digest,
            blob_finalized=blob_finalized,
            skipped_swa_blocks=skipped_swa,
        )

    def finalize_slot_blob(
        self,
        *,
        scope: str,
        slot_id: int,
        blob_path: str | None = None,
    ) -> str | None:
        """Publish deferred slot blob and attach digests (vLLM finish-time store).

        Call after disk save completes or before slot reuse when a finalize is
        pending. Returns digest or None.
        """
        if slot_id < 0:
            return None
        sid = int(slot_id)
        path = blob_path
        with _pending_lock:
            if not path:
                path = _pending_blob_finalize.get((scope, sid))
        if not path:
            return None
        if not Path(path).is_file():
            return None
        if not lmcache_tier_enabled():
            return None
        from runtime.kv.lmcache_blob import publish_slot_blob

        digest = publish_slot_blob(path)
        if not digest:
            return None
        now = time.time() * 1000
        tier = lmcache_tier()
        with self._lock:
            for bh, entry in list(self._blocks.items()):
                if entry.model_scope != scope:
                    continue
                holders = _holder_slots(entry)
                if sid not in holders:
                    continue
                updated = replace(
                    entry,
                    blob_path=path,
                    blob_digest=digest,
                    updated_at_ms=now,
                )
                self._blocks[bh] = updated
                if lmcache_tier_enabled():
                    tier.put(
                        LMCacheBlockRecord(
                            block_hash=bh,
                            parent_hash=entry.parent_hash,
                            block_index=entry.block_index,
                            token_end=entry.token_end,
                            model_scope=scope,
                            session_key=entry.session_key,
                            slot_id=sid,
                            blob_path=path,
                            blob_digest=digest,
                            updated_at_ms=int(now),
                        )
                    )
        with _pending_lock:
            _pending_blob_finalize.pop((scope, sid), None)
        return digest

    def flush_pending_blob_before_reuse(
        self,
        *,
        scope: str,
        slot_id: int,
        blob_path: str | None = None,
    ) -> str | None:
        """Reuse-race flush: finalize before another request claims ``slot_id``."""
        return self.finalize_slot_blob(
            scope=scope, slot_id=slot_id, blob_path=blob_path
        )

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
            blob_digest=rec.blob_digest,
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
        with_digest = sum(1 for e in entries if e.blob_digest)
        from runtime.kv.lmcache_blob import blob_store_health

        return {
            "enabled": prefix_block_pool_enabled(),
            "block_size": prefix_cache_block_size(),
            "max_entries": self.max_entries,
            "entry_count": len(entries),
            "session_count": len(sessions),
            "slot_count": len(slots),
            "multi_holder_blocks": multi_holder,
            "blob_digest_blocks": with_digest,
            # L3-R9 / LA13: capped sample for fleet content-hash routing (newest first).
            "block_hashes": _sample_block_hashes(entries),
            # L3-R11: digests that can be HTTP-pulled (pairs with block_hashes when present).
            "blob_digests": _sample_blob_digests(entries),
            "lmcache_tier": lmcache_tier().health(),
            "lmcache_blobs": blob_store_health(),
        }

    def snapshot_entries(self, *, scope: str | None = None) -> list[PrefixBlockEntry]:
        with self._lock:
            entries = list(self._blocks.values())
        if scope is not None:
            return [e for e in entries if e.model_scope == scope]
        return entries


_POOLS_LOCK = threading.Lock()
_POOLS: dict[str, PrefixBlockPool] = {}


def get_prefix_block_pool(*, model_scope: str) -> PrefixBlockPool:
    with _POOLS_LOCK:
        pool = _POOLS.get(model_scope)
        if pool is None:
            pool = PrefixBlockPool(max_entries=prefix_block_pool_max_entries())
            _POOLS[model_scope] = pool
        return pool


def _radix_health_hash_cap() -> int:
    raw = (os.environ.get("ZEROLLAMA_RADIX_HEALTH_HASH_CAP") or "").strip()
    if not raw:
        return 128
    try:
        return max(0, min(1024, int(raw)))
    except ValueError:
        return 128


def _sample_block_hashes(entries: list[PrefixBlockEntry], *, cap: int | None = None) -> list[str]:
    """Newest-first unique block hashes for /health → fleet LA13 matching."""
    limit = _radix_health_hash_cap() if cap is None else cap
    if limit <= 0 or not entries:
        return []
    ordered = sorted(entries, key=lambda e: e.updated_at_ms, reverse=True)
    out: list[str] = []
    seen: set[str] = set()
    for entry in ordered:
        bh = str(entry.block_hash or "").strip()
        if not bh or bh in seen:
            continue
        seen.add(bh)
        out.append(bh)
        if len(out) >= limit:
            break
    return out


def _sample_blob_digests(entries: list[PrefixBlockEntry], *, cap: int | None = None) -> list[str]:
    """Newest-first unique blob digests for fleet / peer-pull hints."""
    limit = _radix_health_hash_cap() if cap is None else cap
    if limit <= 0 or not entries:
        return []
    ordered = sorted(entries, key=lambda e: e.updated_at_ms, reverse=True)
    out: list[str] = []
    seen: set[str] = set()
    for entry in ordered:
        dig = str(entry.blob_digest or "").strip().lower()
        if not dig or dig in seen:
            continue
        seen.add(dig)
        out.append(dig)
        if len(out) >= limit:
            break
    return out


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
    digest_blocks = 0
    sampled: list[PrefixBlockEntry] = []
    for scope in scopes:
        pool = get_prefix_block_pool(model_scope=scope)
        h = pool.health(scope=scope)
        total += h["entry_count"]
        multi += h.get("multi_holder_blocks", 0)
        digest_blocks += int(h.get("blob_digest_blocks") or 0)
        sampled.extend(pool.snapshot_entries(scope=scope))
    return {
        "enabled": True,
        "block_size": prefix_cache_block_size(),
        "scope_count": len(scopes),
        "entry_count": total,
        "multi_holder_blocks": multi,
        "blob_digest_blocks": digest_blocks,
        "block_hashes": _sample_block_hashes(sampled),
        "blob_digests": _sample_blob_digests(sampled),
        "lmcache_tier": lmcache_tier().health(),
    }


def reset_prefix_block_pools_for_tests() -> None:
    with _POOLS_LOCK:
        _POOLS.clear()
    reset_pending_blob_finalize_for_tests()


def build_model_scope(
    *,
    model_hash: str,
    cache_salt: str | None = None,
) -> str:
    return model_scope_key(model_hash=model_hash, cache_salt=cache_salt)
