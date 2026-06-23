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
    if policy is None:
        spec = resolve_kv_cache_spec()
    else:
        spec = policy.to_spec()
    return spec.cache_prompt_allowed(
        PrefixCacheRequest(
            prompt_cache_key=prompt_cache_key,
            seq_pos=seq_pos,
            prompt_tokens=prompt_tokens,
        ),
        cache_enabled=llama_cache_enabled(),
    )


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
