"""Pluggable KV cache specs (vLLM KVCacheSpec-inspired, GGUF-first).

Each spec describes prefix retention for one GGUF attention layout.
``prefix_cache_policy.py`` builds runtime policy from these specs; Phase 15
native KV can attach allocator hints here without rewriting L3 guards.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runtime.cache_bridge import (
    DEFAULT_CACHE_TTLS,
    default_slot_ttl_ms,
    llama_cache_enabled,
)
from runtime.gguf_estimate import GgufArchHints, gguf_arch_hints
from runtime.kv.hybrid_kv_coordinator import HybridKVCacheCoordinator, build_hybrid_kv_coordinator
from runtime.speculative.plugins import resolve_method

KVCacheKind = str  # standard | sliding_window | hybrid | disabled


def prefix_cache_block_size() -> int:
    """Logical prefix block size for EAGLE drop-last-block (vLLM ``block_size`` analog)."""
    from runtime.env import prefix_cache_block_size as _size

    return _size()


def prefix_cache_retention_interval() -> int | None:
    """SWA sparse retention (``VLLM_PREFIX_CACHE_RETENTION_INTERVAL`` analog).

    Unset env + no YAML → ``0`` (block-aligned checkpoints only; vLLM #52216).
    ``0`` — same (no mid-sequence resume except block boundaries).
    ``>0`` — allow resume only when ``seq_pos`` aligns to this token interval.
    """
    from runtime.env import prefix_cache_retention_interval as _interval

    return _interval()


def drop_last_prefix_block(resume_pos: int, *, block_size: int | None = None) -> int:
    """Drop the last matched prefix block so draft heads recompute (vLLM ``drop_eagle_block``)."""
    pos = max(0, int(resume_pos))
    if pos <= 0:
        return pos
    bs = max(1, int(block_size or prefix_cache_block_size()))
    # seq_pos is next write index; last occupied block starts at this boundary.
    last_block_start = ((pos - 1) // bs) * bs
    return max(0, last_block_start)


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
    drop_last_block_on_resume: bool = False
    extra_retained_tokens: int = 0
    retention_interval: int | None = None
    coordinator: HybridKVCacheCoordinator | None = None
    notes: tuple[str, ...] = ()

    def swa_retention_allows_cache(
        self,
        *,
        seq_pos: int | None = None,
        prompt_tokens: int | None = None,
    ) -> bool:
        """Sparse SWA checkpoint policy (vLLM ``retention_interval``)."""
        if self.coordinator is not None:
            return True
        if self.kind not in ("sliding_window", "hybrid") or self.retention_interval is None:
            return True
        pos = max(0, seq_pos or 0)
        n_prompt = max(0, prompt_tokens or 0)
        if pos == 0:
            return True
        interval = self.retention_interval
        if interval == 0:
            bs = max(1, prefix_cache_block_size())
            if pos % bs == 0:
                return True
            if n_prompt > 0 and (pos + n_prompt) % bs == 0:
                return True
            return False
        if interval <= 0:
            return True
        if pos % interval == 0:
            return True
        if n_prompt > 0 and (pos + n_prompt) % interval == 0:
            return True
        return False

    def swa_allows_cache_prompt(
        self,
        *,
        seq_pos: int | None = None,
        prompt_tokens: int | None = None,
    ) -> bool:
        """SWA/hybrid window guard via coordinator or effective_window."""
        if self.coordinator is not None:
            return self.coordinator.allows_cache_prompt(
                seq_pos=seq_pos,
                prompt_tokens=prompt_tokens,
            )
        if self.kind not in ("sliding_window", "hybrid") or self.effective_window is None:
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
        return self.swa_retention_allows_cache(seq_pos=pos, prompt_tokens=n_prompt)

    def adjust_resume_pos(self, resume_pos: int | None) -> int | None:
        """Apply draft drop-last-block after policy approved ``resume_pos``."""
        if resume_pos is None:
            return None
        pos = int(resume_pos)
        if pos <= 0 or not self.drop_last_block_on_resume:
            return pos
        return drop_last_prefix_block(pos)

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
        return self.adjust_resume_pos(pos)

    def to_health(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind,
            "allow_cache_prompt": self.allow_cache_prompt_base,
            "allow_disk_persist": self.allow_disk_persist,
            "effective_window": self.effective_window,
            "disk_ttl_ms": self.disk_ttl_ms,
            "speculative_draft": self.speculative_draft,
            "drop_last_block_on_resume": self.drop_last_block_on_resume,
            "extra_retained_tokens": self.extra_retained_tokens,
            "retention_interval": self.retention_interval,
            "prefix_cache_block_size": prefix_cache_block_size(),
            "notes": list(self.notes),
        }
        if self.coordinator is not None:
            out["hybrid_coordinator"] = self.coordinator.to_health()
        return out


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
    """Independent or EAGLE/MTP draft model — unsafe for disk slot blobs."""
    llama_type = resolve_method(spec_method or "none")
    if llama_type == "dflash":
        return True
    return llama_type.startswith("draft")


def eagle_block_drop_active(spec_method: str) -> bool:
    """True when last-block drop is required (vLLM ``use_eagle_block_drop``).

    EAGLE/MTP write prefill-lookahead tokens into the last KV block. Independent
    draft models (``draft-simple``, ``dflash``) do not, so they keep full prefix
    hits. vLLM #61455 tags groups that actually contain draft layers; without
    GGUF draft-layer names we still apply this method-level split.
    """
    llama_type = resolve_method(spec_method or "none")
    if llama_type in ("draft-eagle3", "eagle3", "draft-mtp", "mtp"):
        return True
    return llama_type.startswith("draft-eagle") or llama_type.startswith("draft-mtp")


def eagle_extra_retained_tokens(
    spec_method: str, draft_n_max: int | None = None
) -> int:
    """SWA extra tail tokens for multi-token MTP lookahead (vLLM #61455)."""
    if not eagle_block_drop_active(spec_method):
        return 0
    return max(0, int(draft_n_max or 0) - 1)


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
    draft_n_max: int | None = None,
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
    eagle_drop = eagle_block_drop_active(spec_method)
    extra_retained = eagle_extra_retained_tokens(spec_method, draft_n_max)
    retention = prefix_cache_retention_interval()
    coordinator = build_hybrid_kv_coordinator(
        arch,
        num_ctx,
        retention_interval=retention,
        eagle_drop=eagle_drop,
    )
    notes: list[str] = []
    if spec_draft:
        notes.append("disk_disabled_draft_speculative")
    if eagle_drop:
        notes.append("drop_last_block_on_resume_eagle")
        notes.append("drop_last_block_on_resume_draft_speculative")
        if coordinator.eagle_group_count:
            notes.append(f"eagle_groups={coordinator.eagle_group_count}")
    if extra_retained:
        notes.append(f"eagle_extra_retained_tokens={extra_retained}")
    if kind in ("sliding_window", "hybrid"):
        notes.append(f"swa_kind={kind}")
        if window is not None:
            notes.append(f"swa_effective_window={window}")
        if coordinator.full_layer_count:
            notes.append(f"hybrid_full_layers={coordinator.full_layer_count}")
        if coordinator.swa_layer_count:
            notes.append(f"hybrid_swa_layers={coordinator.swa_layer_count}")
    if coordinator.retention_interval is not None:
        notes.append(f"swa_retention_interval={coordinator.retention_interval}")

    return KVCacheSpec(
        kind=kind,
        effective_window=window,
        allow_cache_prompt_base=True,
        allow_disk_persist=not spec_draft,
        disk_ttl_ms=_disk_ttl_for_kind(kind),
        speculative_draft=spec_draft,
        drop_last_block_on_resume=eagle_drop,
        extra_retained_tokens=extra_retained,
        retention_interval=coordinator.retention_interval,
        coordinator=coordinator,
        notes=tuple(notes),
    )
