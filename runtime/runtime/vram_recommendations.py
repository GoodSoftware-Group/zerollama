"""Shared VRAM operator recommendation rules (Phase 13 health + snapshot)."""

from __future__ import annotations

from typing import Any


def skip_global_vram_factor_export(
    *,
    autotune_enabled: bool,
    catalog: list[Any] | None = None,
    factor_source: str | None = None,
    persisted_factor: Any = None,
) -> bool:
    """True when per-GGUF autotune makes a global VRAM_ESTIMATE_FACTOR export misleading."""
    return bool(autotune_enabled) and (
        bool(catalog)
        or factor_source in ("catalog", "session")
        or persisted_factor is not None
    )
