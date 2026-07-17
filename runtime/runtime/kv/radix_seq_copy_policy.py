"""Radix seq-copy admission for hybrid / SWA GGUF layouts (L3-R5).

WHY separate from ``find_radix_share_plan``:
  Block-pool planning is layout-agnostic — it only needs hash chains and slot ids.
  ``llama_memory_seq_cp`` safety depends on the underlying llama.cpp memory type:
  Gemma-style hybrid (full + SWA layers) is safe when the copied prefix fits the
  coordinated SWA window; true attn+recurrent memory (e.g. some LFM2 paths) can
  abort or corrupt logits and stays behind ``ZEROLLAMA_RADIX_HYBRID_SEQ_COPY=0``.

WHY not blanket-skip hybrid like v1:
  ``KVCacheSpec.kind == hybrid`` classifies GGUF full+SWA layouts, not only
  ``llama_memory_hybrid`` attn+recurrent — blocking all hybrid denied Radix on
  common agent models (Gemma) with no upstream bug.

WHY window-only (not full ``swa_allows_cache_prompt`` / retention):
  ``ZEROLLAMA_PREFIX_CACHE_RETENTION_INTERVAL`` sparsifies *same-slot resume*
  checkpoints. Radix copies live dense donor KV verified by the block pool.
  Applying retention here denied warm catch-up at non-aligned ``seq_pos`` and
  fights shared-system-prompt seed under selective retention — same class of
  footgun as vLLM #47782 (Marconi + hybrid retention).
"""

from __future__ import annotations

from runtime.env import radix_hybrid_seq_copy_enabled
from runtime.kv.radix_prefix_share import RadixSharePlan
from runtime.kv_cache_spec import KVCacheSpec


def radix_seq_copy_allowed(
    spec: KVCacheSpec,
    plan: RadixSharePlan,
) -> tuple[bool, str | None]:
    """Return ``(allowed, skip_reason)`` for donor→target KV copy.

    Skip reasons (trace ``radix_seed.skipped``):
      ``hybrid_seq_copy_disabled`` — operator kill-switch for attn+recurrent probes.
      ``hybrid_missing_effective_window`` — hybrid without a resolved SWA window.
      ``hybrid_target_past_swa_window`` — warm target already past SWA window.
      ``hybrid_prefix_exceeds_swa_window`` — copy longer than ``effective_window``.
    """
    if spec.kind in ("standard", "sliding_window"):
        return True, None
    if spec.kind == "disabled":
        return False, "llama_cache_disabled"
    if spec.kind != "hybrid":
        return False, f"unsupported_kv_kind_{spec.kind}"

    if not radix_hybrid_seq_copy_enabled():
        return False, "hybrid_seq_copy_disabled"

    window = spec.effective_window
    if window is None or window <= 0:
        return False, "hybrid_missing_effective_window"

    pos = max(0, int(plan.target_seq_pos_before))
    if pos >= window:
        return False, "hybrid_target_past_swa_window"

    if plan.copy_tokens > window:
        return False, "hybrid_prefix_exceeds_swa_window"

    return True, None
