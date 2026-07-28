"""SWA/hybrid-aware L3 prefix cache policy (vLLM selective-retention inspired).

WHY separate from cache_bridge.py
---------------------------------
Key derivation and slot argv stay in cache_bridge; this module decides *when*
RAM ``cache_prompt`` and disk slot blobs are safe for a given GGUF architecture
and speculative-decode configuration.

Implementation delegates to ``kv_cache_spec.py`` (pluggable ``KVCacheSpec``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runtime.cache_bridge import llama_cache_enabled
from runtime.kv_cache_spec import (
    KVCacheKind,
    KVCacheSpec,
    PrefixCacheRequest,
    classify_gguf_prefix_cache,
    draft_speculative_active,
    effective_prefix_window,
    resolve_kv_cache_spec,
)

PrefixCacheKind = KVCacheKind


@dataclass(frozen=True)
class PrefixCachePolicy:
    kind: PrefixCacheKind
    allow_cache_prompt: bool
    allow_disk_persist: bool
    effective_window: int | None
    disk_ttl_ms: int
    speculative_draft: bool
    drop_last_block_on_resume: bool = False
    retention_interval: int | None = None
    notes: tuple[str, ...] = ()

    @classmethod
    def from_spec(cls, spec: KVCacheSpec) -> PrefixCachePolicy:
        return cls(
            kind=spec.kind,
            allow_cache_prompt=spec.allow_cache_prompt_base,
            allow_disk_persist=spec.allow_disk_persist,
            effective_window=spec.effective_window,
            disk_ttl_ms=spec.disk_ttl_ms,
            speculative_draft=spec.speculative_draft,
            drop_last_block_on_resume=spec.drop_last_block_on_resume,
            retention_interval=spec.retention_interval,
            notes=spec.notes,
        )

    def to_spec(self) -> KVCacheSpec:
        return KVCacheSpec(
            kind=self.kind,
            effective_window=self.effective_window,
            allow_cache_prompt_base=self.allow_cache_prompt,
            allow_disk_persist=self.allow_disk_persist,
            disk_ttl_ms=self.disk_ttl_ms,
            speculative_draft=self.speculative_draft,
            drop_last_block_on_resume=self.drop_last_block_on_resume,
            retention_interval=self.retention_interval,
            notes=self.notes,
        )


def spec_method_from_hints(hints: Any) -> str:
    """Map parsed llama-server argv hints to a policy ``spec_method`` string."""
    st = getattr(hints, "spec_type", None)
    if isinstance(st, str) and st.strip():
        return st.strip()
    return "none"


def swa_cache_prompt_allowed(
    policy: PrefixCachePolicy,
    *,
    seq_pos: int | None = None,
    prompt_tokens: int | None = None,
) -> bool:
    """Whether SWA retention allows ``cache_prompt`` for this request."""
    return policy.to_spec().swa_allows_cache_prompt(
        seq_pos=seq_pos, prompt_tokens=prompt_tokens
    )


def resolve_prefix_cache_policy(
    *,
    gguf: Path | None = None,
    num_ctx: int | None = None,
    spec_method: str = "none",
) -> PrefixCachePolicy:
    """Policy for RAM cache_prompt and disk slot persistence."""
    return PrefixCachePolicy.from_spec(
        resolve_kv_cache_spec(gguf=gguf, num_ctx=num_ctx, spec_method=spec_method)
    )


def cache_prompt_for_request(
    prompt_cache_key: str | None,
    policy: PrefixCachePolicy | None = None,
    *,
    seq_pos: int | None = None,
    prompt_tokens: int | None = None,
) -> bool:
    allow, _, _ = prefix_cache_decision(
        prompt_cache_key,
        policy,
        seq_pos=seq_pos,
        prompt_tokens=prompt_tokens,
    )
    return allow


def _apply_prefix_block_pool(
    allow: bool,
    resume: int | None,
    *,
    prompt_token_ids: list[int] | None,
    model_hash: str | None,
    cache_salt: str | None,
    seq_pos: int | None,
    load_tier_filter: Any = None,
) -> tuple[bool, int | None, str | None]:
    """Hash-chain verification for pinned prefix reuse (optional env)."""
    from runtime.kv.prefix_block_pool import (
        build_model_scope,
        get_prefix_block_pool,
        prefix_block_pool_enabled,
    )

    if not allow or not prefix_block_pool_enabled():
        return allow, resume, None
    if not prompt_token_ids or not model_hash:
        return allow, resume, None
    verify_pos = resume if resume is not None else seq_pos
    if verify_pos is None or verify_pos <= 0:
        return allow, resume, None

    scope = build_model_scope(model_hash=model_hash, cache_salt=cache_salt)
    pool = get_prefix_block_pool(model_scope=scope)
    match = pool.lookup_longest_prefix(
        prompt_token_ids,
        scope=scope,
        seq_pos=verify_pos,
        load_tier_filter=load_tier_filter,
    )
    from runtime.kv_cache_spec import prefix_cache_block_size

    bs = max(1, prefix_cache_block_size())
    expected_blocks = verify_pos // bs
    if expected_blocks > 0 and match.matched_blocks < expected_blocks:
        return False, None, "prefix_block_hash_mismatch"
    return allow, resume, None


def prefix_block_pool_snapshot(
    *,
    prompt_token_ids: list[int] | None,
    model_hash: str | None,
    cache_salt: str | None,
    seq_pos: int | None,
    resume: int | None,
    load_tier_filter: Any = None,
) -> dict[str, Any] | None:
    """Operator/trace snapshot of hash-chain prefix verification."""
    from runtime.kv.prefix_block_pool import (
        build_model_scope,
        get_prefix_block_pool,
        prefix_block_pool_enabled,
    )

    if not prefix_block_pool_enabled():
        return None
    if not prompt_token_ids or not model_hash:
        return {"enabled": True, "skipped": "missing_tokens_or_model_hash"}
    verify_pos = resume if resume is not None else seq_pos
    if verify_pos is None or verify_pos <= 0:
        return {"enabled": True, "skipped": "cold_start"}
    scope = build_model_scope(model_hash=model_hash, cache_salt=cache_salt)
    match = get_prefix_block_pool(model_scope=scope).lookup_longest_prefix(
        prompt_token_ids,
        scope=scope,
        seq_pos=verify_pos,
        load_tier_filter=load_tier_filter,
    )
    from runtime.kv_cache_spec import prefix_cache_block_size

    bs = max(1, prefix_cache_block_size())
    expected_blocks = verify_pos // bs
    return {
        "enabled": True,
        "matched_tokens": match.matched_tokens,
        "matched_blocks": match.matched_blocks,
        "expected_blocks": expected_blocks,
        "lmcache_hits": match.lmcache_hits,
        "verified": expected_blocks == 0 or match.matched_blocks >= expected_blocks,
    }


def prefix_cache_decision(
    prompt_cache_key: str | None,
    policy: PrefixCachePolicy | None = None,
    *,
    seq_pos: int | None = None,
    prompt_tokens: int | None = None,
    prompt_token_ids: list[int] | None = None,
    model_hash: str | None = None,
    cache_salt: str | None = None,
    subprocess: bool = False,
    load_tier_filter: Any = None,
) -> tuple[bool, int | None, str | None]:
    """Return ``(cache_prompt, resume_pos, deny_reason)`` for one admission.

    WHY ``subprocess`` flag: llama-server cannot trim the last KV block (vLLM
    ``drop_eagle_block``). When draft spec would drop the last block in-process,
    subprocess falls back to ``cache_prompt=false`` for correctness.
    """
    if policy is None:
        spec = resolve_kv_cache_spec()
    else:
        spec = policy.to_spec()
    req = PrefixCacheRequest(
        prompt_cache_key=prompt_cache_key,
        seq_pos=seq_pos,
        prompt_tokens=prompt_tokens,
    )
    allow = spec.cache_prompt_allowed(req, cache_enabled=llama_cache_enabled())
    resume = spec.resume_pos(req, cache_enabled=llama_cache_enabled())
    pos = max(0, seq_pos or 0)
    if (
        subprocess
        and spec.drop_last_block_on_resume
        and resume is not None
        and pos > 0
        and resume < pos
    ):
        return False, None, "subprocess_drop_last_block_unsupported"
    return _apply_prefix_block_pool(
        allow,
        resume,
        prompt_token_ids=prompt_token_ids,
        model_hash=model_hash,
        cache_salt=cache_salt,
        seq_pos=seq_pos,
        load_tier_filter=load_tier_filter,
    )


def decode_graph_invalidation_reason(
    *,
    allow: bool,
    resume: int | None,
    seq_pos: int | None,
    slot_pinned: bool,
    deny_reason: str | None = None,
) -> str | None:
    """Reason string for ``bump_decode_graph_epoch`` when prefix KV is invalidated.

    WHY separate from ``prefix_cache_decision``: policy decides ``cache_prompt``;
    this maps the outcome to a trace/invalidation reason for epoch + ggml clear.
    Subprocess uses the reason to POST ``/cuda-graph/invalidate`` before decode.
    """
    if not slot_pinned:
        return None
    pos = max(0, seq_pos or 0)
    if pos <= 0:
        return None
    if deny_reason == "subprocess_drop_last_block_unsupported":
        return "subprocess_drop_last_block"
    if deny_reason == "prefix_block_hash_mismatch":
        return "prefix_block_hash_mismatch"
    if not allow:
        return "cache_prompt_disabled"
    if resume is not None and resume < pos:
        return "drop_last_prefix_block"
    return None


def effective_disk_cache_enabled(
    policy: PrefixCachePolicy | None = None,
) -> bool:
    from runtime.cache_bridge import inprocess_disk_cache_enabled

    if not inprocess_disk_cache_enabled():
        return False
    if policy is not None and not policy.allow_disk_persist:
        return False
    return True


def policy_to_health(policy: PrefixCachePolicy) -> dict[str, Any]:
    return policy.to_spec().to_health()
