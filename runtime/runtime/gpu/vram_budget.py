"""Operator VRAM ceiling (LocalAI LA19 / ZEROLLAMA_VRAM_BUDGET).

Cap is min(detected, budget). Unset leaves probes unchanged. Percentage above
100%% is invalid; absolute values above physical VRAM clamp to detected total.
"""

from __future__ import annotations

import os


def parse_vram_budget(raw: str) -> tuple[float, int]:
    """Return (fraction, absolute_bytes). Both zero means unset.

    fraction in (0, 1]; absolute_bytes when size form. Raises ValueError on
    invalid input. Empty / 0% returns (0, 0).
    """
    s = (raw or "").strip()
    if not s:
        return 0.0, 0
    upper = s.upper()
    if upper.endswith("%"):
        v = float(upper[:-1].strip())
        frac = v / 100.0
        if frac <= 0:
            return 0.0, 0
        if frac > 1:
            raise ValueError(f"vram budget {raw!r} exceeds 100%")
        return frac, 0
    suffixes = (
        ("KIB", 1 << 10),
        ("MIB", 1 << 20),
        ("GIB", 1 << 30),
        ("TIB", 1 << 40),
        ("KB", 1000),
        ("MB", 1000 * 1000),
        ("GB", 1000 * 1000 * 1000),
        ("TB", 1000 * 1000 * 1000 * 1000),
        ("B", 1),
    )
    for suf, mult in suffixes:
        if upper.endswith(suf):
            num = float(upper[: -len(suf)].strip())
            if num < 0:
                raise ValueError(f"invalid vram budget {raw!r}")
            return 0.0, int(num * mult)
    v = float(s)
    if v < 0:
        raise ValueError(f"invalid vram budget {raw!r}: negative")
    if 0 < v <= 1:
        return v, 0
    if v != int(v):
        raise ValueError(f"invalid vram budget {raw!r}: out of range")
    return 0.0, int(v)


def apply_vram_budget(detected_total: int, detected_free: int) -> tuple[int, int]:
    raw = os.environ.get("ZEROLLAMA_VRAM_BUDGET", "").strip()
    if not raw or detected_total <= 0:
        return detected_total, detected_free
    try:
        frac, abs_bytes = parse_vram_budget(raw)
    except ValueError:
        return detected_total, detected_free
    if frac <= 0 and abs_bytes <= 0:
        return detected_total, detected_free
    if frac > 0:
        ceil = int(detected_total * frac)
    else:
        ceil = abs_bytes
    if ceil > detected_total:
        ceil = detected_total
    return min(detected_total, ceil), min(detected_free, ceil)
