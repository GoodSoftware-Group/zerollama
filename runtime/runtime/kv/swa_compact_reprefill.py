"""SWA compact keep-token *budget* helper (DeepInfer YOCO analogue — DI4).

DeepInfer ``KEEP_TOKENS >= window + max_backtrack`` sizes the consumer-SWA
visible/store suffix under ``YOCO_PREFILL_TRUNCATION_MODE=compact``. That path
is **layer-aware**: full/prefill layers still see the long context.

Zerollama already:
- gates resume via ``HybridKVCacheCoordinator``
- filters store via ``swa_reachable_store_mask``

This module only computes the numeric budget. It is **not** a license to slice
the whole prompt before ``llama_decode`` / ``run_prefill`` on hybrid GGUF —
that would drop prefix for full layers. Engine wire for hybrid is a **non-goal**
until ggml can rebuild consumer SWA without truncating full-layer KV (see
``docs/deepinfer-ladder.md`` DI4).
"""

from __future__ import annotations


def compact_reprefill_keep_tokens(
    *,
    sliding_window: int,
    draft_backtrack: int = 0,
) -> int:
    """Keep-token floor: ``window + draft_backtrack`` (DeepInfer KEEP_TOKENS)."""
    window = max(0, int(sliding_window))
    backtrack = max(0, int(draft_backtrack))
    return window + backtrack


def compact_reprefill_query_len(
    prompt_tokens: int,
    *,
    sliding_window: int,
    draft_backtrack: int = 0,
) -> int:
    """Budget for a *consumer-SWA* recompute suffix length.

    ``min(prompt_tokens, window + draft_backtrack)``. When ``sliding_window``
    is unset/non-positive, returns the full prompt length (no compact budget).

    Callers must not apply this as a whole-prompt truncate on hybrid models.
    """
    n = max(0, int(prompt_tokens))
    window = int(sliding_window)
    if window <= 0:
        return n
    keep = compact_reprefill_keep_tokens(
        sliding_window=window,
        draft_backtrack=draft_backtrack,
    )
    return min(n, keep)


def compact_reprefill_suffix_start(
    prompt_tokens: int,
    *,
    sliding_window: int,
    draft_backtrack: int = 0,
) -> int:
    """Start index of a suffix of length ``compact_reprefill_query_len`` (0-based).

    Useful for tests / SWA-only experiments. Do not feed this into hybrid
    full-layer prefill without a layer-safe path.
    """
    n = max(0, int(prompt_tokens))
    q = compact_reprefill_query_len(
        n,
        sliding_window=sliding_window,
        draft_backtrack=draft_backtrack,
    )
    return max(0, n - q)
