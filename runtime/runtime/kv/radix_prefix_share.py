"""Cross-slot Radix-style prefix sharing (vLLM RadixAttention-inspired, slot-level).

WHY this module exists (not full RadixAttention):
  L3 pins each session to ``hash(key) mod parallel``. Two agents with the same
  system prompt but different keys land on different slots and repeat prefill.
  v1 copies one contiguous donor chain into a target before decode — cold seed or
  warm catch-up (L3-R2) — enough for agent fleets without llama-level shared KV
  pages or a scheduler rewrite.

WHY block pool is required:
  Content-addressed hash chains verify the prompt before copy. Seeding without
  verification would reuse stale KV when clients silently change the system prompt.

WHY seq-copy policy is separate (L3-R5): block-pool planning is layout-agnostic;
Gemma-style hybrid (full+SWA layers) is safe when copy fits the SWA window.
Attn+recurrent layouts keep an env kill-switch until probed live.

Product gaps (physical KV pages, blob federation): docs/radix-prefix-share.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from runtime.env import radix_prefix_share_enabled
from runtime.kv.prefix_block_pool import build_model_scope, get_prefix_block_pool
from runtime.kv_cache_spec import prefix_cache_block_size

# Cumulative approximate bytes moved by successful seq_cp (observability only).
_approx_copy_bytes_total = 0
_approx_copy_tokens_total = 0


@dataclass(frozen=True)
class RadixSharePlan:
    source_slot: int
    target_slot: int
    copy_tokens: int
    matched_blocks: int
    tail_block_hash: str | None
    target_seq_pos_before: int = 0
    warm_catchup: bool = False


def find_radix_share_plan(
    tokens: list[int],
    *,
    target_slot: int,
    model_hash: str,
    cache_salt: str | None = None,
    seq_pos: int | None = None,
    effective_window: int | None = None,
) -> RadixSharePlan | None:
    """Plan donor→target KV seed for cold slots or warm catch-up behind donor.

    Cold (``seq_pos == 0``): copy full matched donor prefix into empty target.

    Warm catch-up (L3-R2): when target already holds verified KV for
    ``[0, seq_pos)`` but donor matched further, copy ``[0, donor_matched)`` —
    seq-copy clears target first (redundant re-copy of shared tail is OK).

    WHY verify target before warm copy: without ``verify_target_slot_prefix``,
    a stale partial slot could seed from a donor whose hash chain diverged earlier.
    """
    if not radix_prefix_share_enabled():
        return None
    if target_slot < 0 or not tokens or not model_hash:
        return None
    target_pos = max(0, int(seq_pos or 0))
    scope = build_model_scope(model_hash=model_hash, cache_salt=cache_salt)
    pool = get_prefix_block_pool(model_scope=scope)

    if target_pos > 0:
        if not pool.verify_target_slot_prefix(
            tokens,
            scope=scope,
            target_slot=target_slot,
            seq_pos=target_pos,
        ):
            return None

    found = pool.find_donor_slot_prefix(
        tokens,
        scope=scope,
        target_slot=target_slot,
        max_tokens=len(tokens),
        exclude_slot=target_slot,
        min_matched=target_pos,
    )
    if found is None:
        return None
    source_slot, matched, blocks = found
    if effective_window is not None and effective_window > 0:
        matched = min(matched, effective_window)
        if matched <= 0:
            return None
        blocks = matched // max(1, prefix_cache_block_size())
    if matched <= target_pos:
        return None
    warm = target_pos > 0
    return RadixSharePlan(
        source_slot=source_slot,
        target_slot=target_slot,
        copy_tokens=matched,
        matched_blocks=blocks,
        tail_block_hash=None,
        target_seq_pos_before=target_pos,
        warm_catchup=warm,
    )


def approx_kv_bytes_per_token(
    gguf: Any | None,
    *,
    num_ctx: int,
    n_gpu_layers: int | None = None,
) -> int | None:
    """Rough bytes/token for Radix seq_cp cost (not ggml-exact)."""
    if gguf is None or not num_ctx or num_ctx <= 0:
        return None
    try:
        from pathlib import Path

        from runtime.gguf_estimate import estimate_kv_cache_bytes, gguf_arch_hints

        hints = gguf_arch_hints(Path(gguf))
        total = estimate_kv_cache_bytes(
            hints, int(num_ctx), n_gpu_layers=n_gpu_layers
        )
    except Exception:
        return None
    if not total or total <= 0:
        return None
    return max(1, int(total) // int(num_ctx))


def record_radix_copy_cost(*, copy_tokens: int, bytes_per_token: int | None) -> int | None:
    """Accumulate approximate seq_cp byte cost; return this copy's approx bytes."""
    global _approx_copy_bytes_total, _approx_copy_tokens_total
    toks = max(0, int(copy_tokens))
    _approx_copy_tokens_total += toks
    if bytes_per_token is None or toks <= 0:
        return None
    approx = toks * max(0, int(bytes_per_token))
    _approx_copy_bytes_total += approx
    return approx


def radix_share_health(*, model_scope: str | None = None) -> dict[str, Any]:
    from runtime.env import kv_unified_enabled, kv_unified_operator_note
    from runtime.kv.radix_seq_copy import seq_cp_mode

    return {
        "enabled": radix_prefix_share_enabled(),
        "requires_prefix_block_pool": True,
        "model_scope": model_scope,
        # WHY approx: page-granular COW would cut buffer copy further; L3-R6b
        # tensor deep-copy already forks full aliased K/V on diverge.
        # path (L3-R6a / v59) already zeros this under kv_unified (seq_cp_mode=metadata).
        "approx_copy_tokens_total": _approx_copy_tokens_total,
        "approx_copy_bytes_total": _approx_copy_bytes_total,
        # v52/v54/v58: metadata = unified stream (no buffer copy); buffer_copy = default.
        "kv_unified": kv_unified_enabled(),
        "seq_cp_mode": seq_cp_mode(),
        "kv_unified_note": kv_unified_operator_note(),
    }
