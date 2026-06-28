"""Cross-slot Radix-style prefix sharing (vLLM RadixAttention-inspired, slot-level).

WHY: L3 pins each session to ``hash(key) mod parallel``, so two agents with the
same system prompt land on different slots and repeat prefill. The prefix block
pool already content-addresses token blocks; this module finds a *donor slot*
holding a matching hash chain and copies KV into the target slot before decode.

Requires prefix block pool (auto-enabled with Radix share or ``n_parallel > 1``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from runtime.env import radix_prefix_share_enabled
from runtime.kv.prefix_block_pool import build_model_scope, get_prefix_block_pool
from runtime.kv_cache_spec import prefix_cache_block_size


@dataclass(frozen=True)
class RadixSharePlan:
    source_slot: int
    target_slot: int
    copy_tokens: int
    matched_blocks: int
    tail_block_hash: str | None


def find_radix_share_plan(
    tokens: list[int],
    *,
    target_slot: int,
    model_hash: str,
    cache_salt: str | None = None,
    seq_pos: int | None = None,
    effective_window: int | None = None,
) -> RadixSharePlan | None:
    """Plan a donor→target KV seed when target is cold or behind shared prefix."""
    if not radix_prefix_share_enabled():
        return None
    if target_slot < 0 or not tokens or not model_hash:
        return None
    if seq_pos is not None and seq_pos > 0:
        return None
    scope = build_model_scope(model_hash=model_hash, cache_salt=cache_salt)
    pool = get_prefix_block_pool(model_scope=scope)
    found = pool.find_donor_slot_prefix(
        tokens,
        scope=scope,
        target_slot=target_slot,
        max_tokens=len(tokens),
    )
    if found is None:
        return None
    source_slot, matched, blocks = found
    if effective_window is not None and effective_window > 0:
        matched = min(matched, effective_window)
        if matched <= 0:
            return None
        blocks = matched // max(1, prefix_cache_block_size())
    return RadixSharePlan(
        source_slot=source_slot,
        target_slot=target_slot,
        copy_tokens=matched,
        matched_blocks=blocks,
        tail_block_hash=None,
    )


def radix_share_health(*, model_scope: str | None = None) -> dict[str, Any]:
    return {
        "enabled": radix_prefix_share_enabled(),
        "requires_prefix_block_pool": True,
        "model_scope": model_scope,
    }
