"""Opt-in live llama seq positions on /health (Phase 15 ops).

When ``llama_parallel_slots==1``, in-process uses per-request contexts and
``kv_physical`` cannot show live ``llama_pos_*``. Setting
``ZEROLLAMA_RUNTIME_KV_LIVE_PHYSICAL=1`` bumps effective ``-np`` to 2 for the
in-process backend only (never overrides explicit ``-np`` in argv).
"""

from __future__ import annotations

import os
from typing import Any

from runtime.llama_args import parse_llama_server_args, resolve_parallel_slots

_LIVE_PHYSICAL_ENV = "ZEROLLAMA_RUNTIME_KV_LIVE_PHYSICAL"
_MIN_LIVE_SLOTS = 2


def kv_live_physical_enabled() -> bool:
    return os.environ.get(_LIVE_PHYSICAL_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def effective_parallel_slots(
    llama_args: list[str] | None,
    *,
    default: int,
    backend: str,
) -> int:
    """Effective ``-np`` for slot allocator and in-process ``n_seq_max``."""
    slots, _ = effective_parallel_slots_detail(
        llama_args, default=default, backend=backend
    )
    return slots


def effective_parallel_slots_detail(
    llama_args: list[str] | None,
    *,
    default: int,
    backend: str,
) -> tuple[int, dict[str, Any]]:
    configured = resolve_parallel_slots(llama_args, default=default)
    parsed = parse_llama_server_args(llama_args or [])
    meta: dict[str, Any] = {
        "enabled": kv_live_physical_enabled(),
        "configured": configured,
        "effective": configured,
        "applied": False,
    }
    if not kv_live_physical_enabled():
        return configured, meta
    if backend != "inprocess":
        meta["reason"] = "inprocess_only"
        return configured, meta
    # Config builds ``-np`` from YAML; only treat argv as explicit when it overrides default.
    if parsed.parallel_slots is not None and parsed.parallel_slots != default:
        meta["reason"] = "explicit_np_in_argv"
        return configured, meta
    if configured >= _MIN_LIVE_SLOTS:
        meta["reason"] = "already_multi_seq"
        return configured, meta
    meta["applied"] = True
    meta["effective"] = _MIN_LIVE_SLOTS
    return _MIN_LIVE_SLOTS, meta


def kv_live_physical_health(
    llama_args: list[str] | None,
    *,
    default: int,
    backend: str,
) -> dict[str, Any]:
    _, detail = effective_parallel_slots_detail(
        llama_args, default=default, backend=backend
    )
    detail["env"] = _LIVE_PHYSICAL_ENV
    detail["min_slots"] = _MIN_LIVE_SLOTS
    if detail.get("applied"):
        detail["note"] = (
            "shared in-process ctx enabled for live kv_physical seq positions"
        )
    elif detail.get("enabled") and detail.get("reason") == "inprocess_only":
        detail["note"] = "set ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess to apply"
    return detail
