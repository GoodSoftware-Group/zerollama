"""DI5 — eval hygiene mode (DeepInfer ``DEEPINFER_BATCH_INVARIANT`` *idea*, not kernels).

DeepInfer disables FlashMLA split-KV and pins per-seq ``seq_seed`` so goldens
do not drift with co-batching. ggml has no split-KV toggle.

``ZEROLLAMA_BATCH_INVARIANT=1`` is an **eval hygiene** switch (default off):

- **Force** greedy sampling (``temperature=0``, ``top_k=1``, ``top_p=1``) —
  overrides client/Modelfile sampler knobs so doctor/golden runs are not
  silently non-greedy when options already contain ``temperature: 0.8``
- Pin ``seed`` to ``ZEROLLAMA_SEQ_SEED`` (default ``0``); client ``seed>=0``
  wins; ``seed=-1`` / missing → process default
- Deny prefix KV reuse (``cache_prompt``) and Radix cross-slot seed
- Skip Phase 15 auto-batch / stream auto-batch
- Serialize ``generate_batch`` to per-prompt ``generate`` (no multi-row decode)

Not a claim of bit-identical CUDA across batch shapes. Prefer lab ports +
``n_parallel=1`` / single-slot serve for the strongest goldens.
"""

from __future__ import annotations

import os
from typing import Any


def batch_invariant_enabled() -> bool:
    from runtime.env import env_bool

    return env_bool("ZEROLLAMA_BATCH_INVARIANT", default=False)


def seq_seed_default() -> int:
    """Process-wide eval seed when request omits a non-negative ``seed``."""
    raw = os.environ.get("ZEROLLAMA_SEQ_SEED", "").strip()
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError:
        return 0


def _coerce_seed(raw: Any) -> int | None:
    """Return non-negative seed, or None if missing / random (``-1``)."""
    if raw is None:
        return None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    if n < 0:
        return None
    return n


def apply_batch_invariant_options(options: dict[str, Any] | None) -> dict[str, Any]:
    """Return options with eval sampling forced (no-op when disabled).

    When enabled, always overrides temperature / top_k / top_p to greedy.
    Seed: keep client ``seed>=0``, else ``ZEROLLAMA_SEQ_SEED``.
    """
    out: dict[str, Any] = dict(options or {})
    if not batch_invariant_enabled():
        return out
    client_seed = _coerce_seed(out.get("seed"))
    out["seed"] = client_seed if client_seed is not None else seq_seed_default()
    out["temperature"] = 0.0
    out["top_k"] = 1
    out["top_p"] = 1.0
    return out


def batch_invariant_denies_prefix_cache() -> bool:
    """True when eval mode must disable cache_prompt / Radix seed."""
    return batch_invariant_enabled()


def batch_invariant_denies_auto_batch() -> bool:
    """True when eval mode must not coalesce concurrent generates."""
    return batch_invariant_enabled()


def batch_invariant_serializes_generate_batch() -> bool:
    """True when ``generate_batch`` must fall back to serial ``generate``."""
    return batch_invariant_enabled()


def batch_invariant_health() -> dict[str, Any]:
    enabled = batch_invariant_enabled()
    return {
        "enabled": enabled,
        "mode": "eval_hygiene" if enabled else "off",
        "seq_seed_default": seq_seed_default() if enabled else None,
        "forces_greedy": enabled,
        "denies_prefix_cache": enabled,
        "denies_auto_batch": enabled,
        "serializes_generate_batch": enabled,
        "note": (
            "eval hygiene — forces greedy+seed, denies KV reuse/auto-batch, "
            "serializes generate_batch; ggml logits may still vary with "
            "n_parallel>1 continuous batch (prefer single-slot lab serve); "
            "no FlashMLA split-KV switch"
            if enabled
            else "set ZEROLLAMA_BATCH_INVARIANT=1 for doctor/golden smokes"
        ),
    }
