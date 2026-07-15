"""Map llama-server completion timings to Ollama-shaped metric fields.

Why dual keys for cache hits: Go subprocess unmarshals CompletionResponse
(prompt_eval_cached_count); HTTP clients expect cached_prompt_tokens on done chunks.
prompt_eval_count = cache_n + prompt_n (total prompt tokens presented to the model).
"""

from __future__ import annotations

from typing import Any


def _ms_to_ns(ms: float) -> int:
    if ms <= 0:
        return 0
    return int(ms * 1_000_000)


def metrics_from_llama_chunk(chunk: dict[str, Any]) -> dict[str, int]:
    """Extract prompt/eval counts and durations from a streaming completion chunk."""
    timings = chunk.get("timings")
    if not isinstance(timings, dict):
        return {}
    cache_n = int(timings.get("cache_n") or 0)
    prompt_n = int(timings.get("prompt_n") or 0)
    predict_n = int(timings.get("predicted_n") or timings.get("predict_n") or 0)
    prompt_ms = float(timings.get("prompt_ms") or 0.0)
    predict_ms = float(timings.get("predicted_ms") or timings.get("predict_ms") or 0.0)
    out: dict[str, int] = {}
    if cache_n > 0:
        out["prompt_eval_cached_count"] = cache_n
        out["cached_prompt_tokens"] = cache_n
    if cache_n + prompt_n > 0:
        out["prompt_eval_count"] = cache_n + prompt_n
    if prompt_ms > 0:
        out["prompt_eval_duration"] = _ms_to_ns(prompt_ms)
    if predict_n > 0:
        out["eval_count"] = predict_n
    if predict_ms > 0:
        out["eval_duration"] = _ms_to_ns(predict_ms)
    return out


def detect_context_overflow(
    metrics: dict[str, Any],
    num_ctx: int | None,
    original_prompt_tokens: int | None,
) -> dict[str, Any]:
    """Return prompt_truncated / original_prompt_tokens when the backend
    silently truncated the prompt via context shift.

    Detection: prompt_eval_count is pinned at or near num_ctx while the
    original prompt was larger.  We also check ``done_reason == "length"``
    as a secondary signal but do not require it.
    """
    if num_ctx is None or num_ctx <= 0:
        return {}
    pec = metrics.get("prompt_eval_count", 0)
    if pec <= 0:
        return {}
    orig = original_prompt_tokens or 0
    if orig <= num_ctx and pec >= num_ctx - 8:
        orig = 0
    if orig > num_ctx:
        return {
            "prompt_truncated": True,
            "original_prompt_tokens": orig,
        }
    if pec >= num_ctx - 8 and orig > pec:
        return {
            "prompt_truncated": True,
            "original_prompt_tokens": orig,
        }
    return {}
