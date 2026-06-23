"""Pluggable KV cache specs (vLLM KVCacheSpec-inspired, GGUF-first).

Each spec describes prefix retention for one GGUF attention layout.
``prefix_cache_policy.py`` builds runtime policy from these specs; Phase 15
native KV can attach allocator hints here without rewriting L3 guards.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runtime.cache_bridge import (
    DEFAULT_CACHE_TTLS,
    default_slot_ttl_ms,
    llama_cache_enabled,
)
from runtime.gguf_estimate import GgufArchHints, gguf_arch_hints
from runtime.speculative.plugins import resolve_method

KVCacheKind = str  # standard | sliding_window | hybrid | disabled


@dataclass(frozen=True)
class PrefixCacheRequest:
    """Per-request inputs for prefix reuse decisions."""

    prompt_cache_key: str | None
    seq_pos: int | None = None
    prompt_tokens: int | None = None


@dataclass(frozen=True)
class KVCacheSpec:
    """Architecture + spec-decode prefix retention contract."""

    kind: KVCacheKind
    effective_window: int | None
    allow_cache_prompt_base: bool
    allow_disk_persist: bool
    disk_ttl_ms: int
    speculative_draft: bool
    notes: tuple[str, ...] = ()

    def swa_allows_cache_prompt(
        self,
        *,
        seq_pos: int | None = None,
        prompt_tokens: int | None = None,
    ) -> bool:
        """SWA window guard; hybrid/standard specs always return True."""
        if self.kind != "sliding_window" or self.effective_window is None:
            return True
        window = self.effective_window
        pos = max(0, seq_pos or 0)
        n_prompt = max(0, prompt_tokens or 0)
        if pos >= window:
            return False
        if n_prompt > window:
            return False
        if pos > 0 and n_prompt > 0 and pos + n_prompt > window:
            return False
        return True

    def cache_prompt_allowed(
        self,
        req: PrefixCacheRequest | None = None,
        *,
        prompt_cache_key: str | None = None,
        seq_pos: int | None = None,
        prompt_tokens: int | None = None,
        cache_enabled: bool | None = None,
    ) -> bool:
        enabled = llama_cache_enabled() if cache_enabled is None else cache_enabled
        key = prompt_cache_key
        pos = seq_pos
        n_prompt = prompt_tokens
        if req is not None:
            key = req.prompt_cache_key
            pos = req.seq_pos if pos is None else pos
            n_prompt = req.prompt_tokens if n_prompt is None else n_prompt
        if not key or not enabled:
            return False
        if not self.allow_cache_prompt_base:
            return False
        return self.swa_allows_cache_prompt(seq_pos=pos, prompt_tokens=n_prompt)

    def resume_pos(
        self,
        req: PrefixCacheRequest | None = None,
        *,
        prompt_cache_key: str | None = None,
        seq_pos: int | None = None,
        prompt_tokens: int | None = None,
        cache_enabled: bool | None = None,
    ) -> int | None:
        """Live seq position for KV resume, or ``None`` when reuse is blocked."""
        pos = seq_pos
        if req is not None and pos is None:
            pos = req.seq_pos
        if not self.cache_prompt_allowed(
            req=req,
            prompt_cache_key=prompt_cache_key,
            seq_pos=pos,
            prompt_tokens=prompt_tokens,
            cache_enabled=cache_enabled,
        ):
            return None
        return pos

    def to_health(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "allow_cache_prompt": self.allow_cache_prompt_base,
            "allow_disk_persist": self.allow_disk_persist,
            "effective_window": self.effective_window,
            "disk_ttl_ms": self.disk_ttl_ms,
            "speculative_draft": self.speculative_draft,
            "notes": list(self.notes),
        }


def classify_gguf_prefix_cache(arch: GgufArchHints) -> KVCacheKind:
    """Classify prefix reuse semantics from GGUF attention metadata."""
    per = arch.sliding_window_per_layer
    layers = arch.scalar.get("block_count") or 0
    if per and layers > 0 and len(per) == layers:
        sw_layers = sum(1 for w in per if w > 0)
        full_layers = sum(1 for w in per if w == 0)
        if sw_layers > 0 and full_layers > 0:
            return "hybrid"
        if sw_layers > 0:
            return "sliding_window"
    sw = arch.scalar.get("sliding_window")
    if sw and sw > 0:
        return "sliding_window"
    return "standard"


def effective_prefix_window(arch: GgufArchHints, num_ctx: int | None) -> int | None:
    """Max tokens whose KV is fully retained for prefix matching on SWA layers."""
    if num_ctx is None or num_ctx <= 0:
        return None
    per = arch.sliding_window_per_layer
    layers = arch.scalar.get("block_count") or 0
    windows: list[int] = []
    per_authoritative = bool(per and layers > 0 and len(per) == layers)
    if per_authoritative:
        for w in per:
            if w > 0:
                windows.append(min(num_ctx, w))
    else:
        sw = arch.scalar.get("sliding_window")
        if sw and sw > 0:
            windows.append(min(num_ctx, sw))
    if not windows:
        return num_ctx
    return min(windows)


def draft_speculative_active(spec_method: str) -> bool:
    """Draft-based spec decode (eagle/mtp/dflash) — unsafe for disk slot blobs."""
    llama_type = resolve_method(spec_method or "none")
    return llama_type.startswith("draft")


def _disk_ttl_for_kind(kind: KVCacheKind) -> int:
    if kind == "sliding_window":
        return DEFAULT_CACHE_TTLS["short"]
    if kind == "hybrid":
        return DEFAULT_CACHE_TTLS["long"]
    return default_slot_ttl_ms()


def resolve_kv_cache_spec(
    *,
    gguf: Path | None = None,
    num_ctx: int | None = None,
    spec_method: str = "none",
) -> KVCacheSpec:
    """Resolve the KV cache spec for a model + speculative configuration."""
    if not llama_cache_enabled():
        return KVCacheSpec(
            kind="disabled",
            effective_window=None,
            allow_cache_prompt_base=False,
            allow_disk_persist=False,
            disk_ttl_ms=default_slot_ttl_ms(),
            speculative_draft=False,
            notes=("llama_cache_disabled",),
        )

    arch = GgufArchHints()
    if gguf is not None and gguf.is_file():
        try:
            arch = gguf_arch_hints(gguf)
        except (OSError, ValueError):
            pass

    kind = classify_gguf_prefix_cache(arch)
    window = effective_prefix_window(arch, num_ctx)
    spec_draft = draft_speculative_active(spec_method)
    notes: list[str] = []
    if spec_draft:
        notes.append("disk_disabled_draft_speculative")
        notes.append("cache_prompt_disabled_draft_speculative")
    if kind in ("sliding_window", "hybrid"):
        notes.append(f"swa_kind={kind}")
        if window is not None:
            notes.append(f"swa_effective_window={window}")

    allow = not spec_draft
    return KVCacheSpec(
        kind=kind,
        effective_window=window,
        allow_cache_prompt_base=allow,
        allow_disk_persist=allow,
        disk_ttl_ms=_disk_ttl_for_kind(kind),
        speculative_draft=spec_draft,
        notes=tuple(notes),
    )
