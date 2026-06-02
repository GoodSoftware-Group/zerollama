"""Resolve GGUF paths from Ollama-shaped request options (Phase 9 manifest → runtime)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _parse_gguf_option(raw: Any) -> Path | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        return None
    p = Path(raw.strip())
    if not p.is_absolute():
        p = p.resolve()
    return p


def peek_gguf_path(options: dict[str, Any]) -> Path | None:
    """Read GGUF path from options without removing keys."""
    for key in ("gguf", "model_path"):
        p = _parse_gguf_option(options.get(key))
        if p is not None:
            return p
    return None


def pop_gguf_path(options: dict[str, Any]) -> Path | None:
    """Remove and return an absolute GGUF path from options, if present."""
    for key in ("gguf", "model_path"):
        raw = options.pop(key, None)
        p = _parse_gguf_option(raw)
        if p is not None:
            return p
    return None
