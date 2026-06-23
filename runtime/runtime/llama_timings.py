"""Map llama-server completion timings to Ollama-shaped metric fields.

Why dual keys for cache hits: Go subprocess unmarshals CompletionResponse
(prompt_eval_cached_count); HTTP clients expect cached_prompt_tokens on done chunks.
prompt_eval_count = cache_n + prompt_n (total prompt tokens presented to the model).
"""

from __future__ import annotations

from typing import Any


def metrics_from_llama_chunk(chunk: dict[str, Any]) -> dict[str, int]:
    """Extract prompt/eval counts from a streaming completion chunk."""
    timings = chunk.get("timings")
    if not isinstance(timings, dict):
        return {}
    cache_n = int(timings.get("cache_n") or 0)
    prompt_n = int(timings.get("prompt_n") or 0)
    predict_n = int(timings.get("predicted_n") or timings.get("predict_n") or 0)
    out: dict[str, int] = {}
    if cache_n > 0:
        out["prompt_eval_cached_count"] = cache_n
        out["cached_prompt_tokens"] = cache_n
    if cache_n+prompt_n > 0:
        out["prompt_eval_count"] = cache_n + prompt_n
    if predict_n > 0:
        out["eval_count"] = predict_n
    return out
