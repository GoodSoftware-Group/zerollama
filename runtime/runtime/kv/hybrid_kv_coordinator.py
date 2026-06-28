"""Hybrid KV cache coordinator (vLLM HybridKVCacheCoordinator-inspired, GGUF-first).

WHY: Hybrid models (Gemma 3/4, etc.) mix full-attention and sliding-window layers.
Prefix reuse must respect the *tightest* SWA window — full layers may retain the
entire prefix in KV, but SWA layers only hold the last ``W`` tokens. Allowing
``cache_prompt`` when ``seq_pos >= W`` would restore slot KV that SWA layers
cannot represent.

This module classifies per-layer groups from GGUF metadata and applies a single
coordinated retention gate for L3 ``cache_prompt`` / resume (not a block pool).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

from runtime.gguf_estimate import GgufArchHints

LayerGroupKind = Literal["full", "sliding_window"]
CoordinatorKind = Literal["standard", "sliding_window", "hybrid"]


@dataclass(frozen=True)
class LayerGroupSpec:
    """One attention layout group (full ctx or SWA window)."""

    kind: LayerGroupKind
    layer_indices: tuple[int, ...]
    window: int | None

    def to_health(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "layers": list(self.layer_indices),
            "layer_count": len(self.layer_indices),
            "window": self.window,
        }


@dataclass(frozen=True)
class HybridKVCacheCoordinator:
    """Coordinated prefix retention across full + SWA layer groups."""

    kind: CoordinatorKind
    layer_groups: tuple[LayerGroupSpec, ...]
    num_layers: int
    num_ctx: int | None
    swa_effective_window: int | None
    retention_interval: int | None = None

    @property
    def full_layer_count(self) -> int:
        return sum(len(g.layer_indices) for g in self.layer_groups if g.kind == "full")

    @property
    def swa_layer_count(self) -> int:
        return sum(
            len(g.layer_indices) for g in self.layer_groups if g.kind == "sliding_window"
        )

    def allows_cache_prompt(
        self,
        *,
        seq_pos: int | None = None,
        prompt_tokens: int | None = None,
    ) -> bool:
        """True when all groups can retain prefix for this admission."""
        if self.swa_effective_window is None:
            return True
        window = self.swa_effective_window
        pos = max(0, seq_pos or 0)
        n_prompt = max(0, prompt_tokens or 0)
        if pos >= window:
            return False
        if n_prompt > window:
            return False
        if pos > 0 and n_prompt > 0 and pos + n_prompt > window:
            return False
        return self._retention_allows(pos=pos, n_prompt=n_prompt)

    def coordinated_resume_pos(self, seq_pos: int | None) -> int | None:
        """Live seq index safe for coordinated resume, or ``None`` when blocked."""
        if seq_pos is None:
            return None
        pos = max(0, int(seq_pos))
        if not self.allows_cache_prompt(seq_pos=pos, prompt_tokens=0):
            return None
        return pos

    def _retention_allows(self, *, pos: int, n_prompt: int) -> bool:
        if self.retention_interval is None:
            return True
        interval = self.retention_interval
        if interval == 0:
            from runtime.kv_cache_spec import prefix_cache_block_size

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

    def to_health(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "num_layers": self.num_layers,
            "num_ctx": self.num_ctx,
            "swa_effective_window": self.swa_effective_window,
            "full_layer_count": self.full_layer_count,
            "swa_layer_count": self.swa_layer_count,
            "retention_interval": self.retention_interval,
            "layer_groups": [g.to_health() for g in self.layer_groups],
        }


def _min_swa_window(groups: tuple[LayerGroupSpec, ...]) -> int | None:
    windows = [g.window for g in groups if g.kind == "sliding_window" and g.window]
    if not windows:
        return None
    return min(windows)


def build_layer_groups(arch: GgufArchHints, num_ctx: int | None) -> tuple[LayerGroupSpec, ...]:
    layers = int(arch.scalar.get("block_count") or 0)
    if layers <= 0:
        return ()

    per = arch.sliding_window_per_layer
    if per and len(per) == layers:
        full_indices: list[int] = []
        swa_by_window: dict[int, list[int]] = defaultdict(list)
        for idx, w in enumerate(per):
            if w > 0:
                cap = min(num_ctx, w) if num_ctx and num_ctx > 0 else w
                swa_by_window[int(cap)].append(idx)
            else:
                full_indices.append(idx)
        groups: list[LayerGroupSpec] = []
        if full_indices:
            groups.append(
                LayerGroupSpec("full", tuple(full_indices), None),
            )
        for window in sorted(swa_by_window):
            indices = tuple(swa_by_window[window])
            groups.append(LayerGroupSpec("sliding_window", indices, window))
        return tuple(groups)

    sw = arch.scalar.get("sliding_window")
    if sw and sw > 0:
        window = min(num_ctx, sw) if num_ctx and num_ctx > 0 else int(sw)
        return (
            LayerGroupSpec(
                "sliding_window",
                tuple(range(layers)),
                window,
            ),
        )
    return (LayerGroupSpec("full", tuple(range(layers)), None),)


def classify_coordinator_kind(groups: tuple[LayerGroupSpec, ...]) -> CoordinatorKind:
    has_full = any(g.kind == "full" for g in groups)
    has_swa = any(g.kind == "sliding_window" for g in groups)
    if has_full and has_swa:
        return "hybrid"
    if has_swa:
        return "sliding_window"
    return "standard"


def build_hybrid_kv_coordinator(
    arch: GgufArchHints,
    num_ctx: int | None,
    *,
    retention_interval: int | None = None,
) -> HybridKVCacheCoordinator:
    groups = build_layer_groups(arch, num_ctx)
    layers = int(arch.scalar.get("block_count") or 0)
    kind = classify_coordinator_kind(groups)
    swa_window = _min_swa_window(groups)
    if kind == "standard":
        swa_window = num_ctx if num_ctx and num_ctx > 0 else None
    retention = retention_interval
    if retention is None:
        from runtime.kv_cache_spec import prefix_cache_retention_interval

        retention = prefix_cache_retention_interval()
    if kind == "standard":
        retention = None
    return HybridKVCacheCoordinator(
        kind=kind,
        layer_groups=groups,
        num_layers=layers,
        num_ctx=num_ctx,
        swa_effective_window=swa_window,
        retention_interval=retention if kind in ("sliding_window", "hybrid") else None,
    )
