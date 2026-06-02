"""Inference request priority (Phase 11), aligned with Go training T6."""

from __future__ import annotations

from enum import Enum


class InferencePriority(str, Enum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


def parse_inference_priority(raw: str | None) -> InferencePriority:
    """Map Ollama-style priority strings to runtime classes."""
    if not raw:
        return InferencePriority.NORMAL
    key = raw.strip().lower()
    if key in ("high", "interactive", "urgent"):
        return InferencePriority.HIGH
    if key in ("low", "batch", "background"):
        return InferencePriority.LOW
    return InferencePriority.NORMAL


def priority_from_options(options: dict | None) -> InferencePriority:
    if not options:
        return InferencePriority.NORMAL
    raw = options.get("priority")
    if raw is None:
        return InferencePriority.NORMAL
    return parse_inference_priority(str(raw))
