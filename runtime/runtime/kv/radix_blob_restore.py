"""L3-R7 — cold-node Radix restore from federated LMCache slot blobs.

When no live donor slot exists for ``seq_cp`` but the prefix block pool (or Redis
tier) has a ``blob_digest``, materialize the content-addressed slot ``.bin`` into
the target slot path and load it with ``llama_state_seq_load_file`` (in-process).
Subprocess path materializes the file under ``--slot-save-path`` for disk restore.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from runtime.env import radix_prefix_share_enabled
from runtime.kv.prefix_block_pool import build_model_scope, get_prefix_block_pool


@dataclass(frozen=True)
class BlobRestorePlan:
    target_slot: int
    restore_tokens: int
    matched_blocks: int
    blob_digest: str
    tail_block_hash: str | None
    source_slot_id: int | None = None


def find_blob_restore_plan(
    tokens: list[int],
    *,
    target_slot: int,
    model_hash: str,
    cache_salt: str | None = None,
    seq_pos: int | None = None,
    effective_window: int | None = None,
    load_tier_filter: Any = None,
) -> BlobRestorePlan | None:
    """Plan cold restore when live Radix donor is absent but a blob digest exists."""
    from runtime.kv.lmcache_blob import lmcache_blobs_enabled

    if not radix_prefix_share_enabled() or not lmcache_blobs_enabled():
        return None
    if target_slot < 0 or not tokens or not model_hash:
        return None
    target_pos = max(0, int(seq_pos or 0))
    # Warm catch-up still prefers live seq_cp; blob restore is cold-slot oriented.
    if target_pos > 0:
        return None
    scope = build_model_scope(model_hash=model_hash, cache_salt=cache_salt)
    pool = get_prefix_block_pool(model_scope=scope)
    found = pool.find_blob_prefix(
        tokens,
        scope=scope,
        max_tokens=len(tokens),
        min_matched=1,
        load_tier_filter=load_tier_filter,
    )
    if found is None:
        return None
    matched = found.matched_tokens
    if effective_window is not None and effective_window > 0:
        matched = min(matched, effective_window)
    if matched <= 0:
        return None
    return BlobRestorePlan(
        target_slot=target_slot,
        restore_tokens=matched,
        matched_blocks=found.matched_blocks,
        blob_digest=found.blob_digest,
        tail_block_hash=found.tail_hash,
        source_slot_id=found.source_slot_id,
    )


def execute_blob_restore_plan(
    plan: BlobRestorePlan,
    *,
    model_hash: str,
    inprocess_lib: Any | None = None,
    inprocess_ctx: Any | None = None,
    token_capacity: int | None = None,
) -> dict[str, Any]:
    """Materialize blob to target slot path; load in-process when lib/ctx given.

    Returns a trace dict with ``restored_tokens`` (0 if materialize/load failed).
    """
    from runtime.cache_bridge import prepare_slot_cache_dir, slot_cache_file_path
    from runtime.kv.lmcache_blob import materialize_blob

    prepare_slot_cache_dir(model_hash)
    dest = slot_cache_file_path(model_hash, plan.target_slot, 0)
    if not materialize_blob(plan.blob_digest, dest):
        return {
            "mode": "blob_restore",
            "ok": False,
            "restored_tokens": 0,
            "skipped": "materialize_failed",
            "blob_digest": plan.blob_digest[:16],
            "target_slot": plan.target_slot,
        }
    restored = 0
    if inprocess_lib is not None and inprocess_ctx is not None:
        from runtime.worker.libllama_ctypes import load_slot_cache_disk_file

        cap = int(token_capacity or plan.restore_tokens or 4096)
        restored = load_slot_cache_disk_file(
            inprocess_lib,
            inprocess_ctx,
            seq_id=plan.target_slot,
            path=dest,
            token_capacity=max(cap, plan.restore_tokens),
        )
    return {
        "mode": "blob_restore",
        "ok": restored > 0 or inprocess_lib is None,
        "restored_tokens": restored if inprocess_lib is not None else plan.restore_tokens,
        "materialized": True,
        "blob_digest": plan.blob_digest[:16],
        "target_slot": plan.target_slot,
        "matched_blocks": plan.matched_blocks,
        "source_slot_id": plan.source_slot_id,
        "path": str(dest),
    }
