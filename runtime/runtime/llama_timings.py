"""Map llama-server completion timings to Ollama-shaped metric fields.

Why dual keys for cache hits: Go subprocess unmarshals CompletionResponse
(prompt_eval_cached_count); HTTP clients expect cached_prompt_tokens on done chunks.
prompt_eval_count = cache_n + prompt_n (total prompt tokens presented to the model).

HiCache tiers (SGLang sglext.cached_tokens_details):
  device  — GPU/slot cache_n (existing cached_prompt_tokens)
  host    — in-process disk slot restore (complete.disk_restore)
  storage — L3 federated blob restore (radix blob / LMCache)
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse


def _ms_to_ns(ms: float) -> int:
    if ms <= 0:
        return 0
    return int(ms * 1_000_000)


def lmcache_storage_backend_label() -> str:
    """Scheme from ZEROLLAMA_LMCACHE_URI / YAML (file, redis, …); default file."""
    try:
        from runtime.env import lmcache_uri

        uri = (lmcache_uri() or "").strip()
    except Exception:
        uri = ""
    if not uri:
        return "file"
    parsed = urlparse(uri)
    if parsed.scheme:
        return parsed.scheme
    return "file"


def merge_cache_tier_details(
    out: dict[str, Any],
    *,
    host: int = 0,
    storage: int = 0,
    storage_backend: str = "",
    creation: int = 0,
) -> dict[str, Any]:
    """Attach optional host/storage/creation tier counts (omit zeros)."""
    if host > 0:
        out["cached_tokens_host"] = int(host)
        out["prompt_eval_cached_host"] = int(host)
    if storage > 0:
        out["cached_tokens_storage"] = int(storage)
        out["prompt_eval_cached_storage"] = int(storage)
        backend = (storage_backend or "").strip() or lmcache_storage_backend_label()
        out["cached_tokens_storage_backend"] = backend
    if creation > 0:
        out["cache_creation_tokens"] = int(creation)
        out["prompt_eval_cache_creation_count"] = int(creation)
        out["created_cache_tokens"] = int(creation)
    return out


def metrics_from_llama_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """Extract prompt/eval counts and durations from a streaming completion chunk."""
    timings = chunk.get("timings")
    out: dict[str, Any] = {}
    if isinstance(timings, dict):
        cache_n = int(timings.get("cache_n") or 0)
        prompt_n = int(timings.get("prompt_n") or 0)
        predict_n = int(timings.get("predicted_n") or timings.get("predict_n") or 0)
        prompt_ms = float(timings.get("prompt_ms") or 0.0)
        predict_ms = float(timings.get("predicted_ms") or timings.get("predict_ms") or 0.0)
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
    # Tier fields may be stamped on the chunk by ctypes/engine (not in timings).
    host = int(chunk.get("cached_tokens_host") or chunk.get("prompt_eval_cached_host") or 0)
    storage = int(
        chunk.get("cached_tokens_storage") or chunk.get("prompt_eval_cached_storage") or 0
    )
    backend = str(chunk.get("cached_tokens_storage_backend") or "")
    creation = int(
        chunk.get("cache_creation_tokens")
        or chunk.get("prompt_eval_cache_creation_count")
        or chunk.get("created_cache_tokens")
        or 0
    )
    merge_cache_tier_details(
        out, host=host, storage=storage, storage_backend=backend, creation=creation
    )
    return out


def detect_context_overflow(
    metrics: dict[str, Any],
    num_ctx: int | None,
    original_prompt_tokens: int | None,
) -> dict[str, Any]:
    """Return ``prompt_truncated`` / ``original_prompt_tokens`` when the backend
    silently truncated the prompt (context shift / slot fill).

    Why this exists: the runtime proxy forwards prompts to llama-server without
    Go-side ``chatPrompt`` truncation. llama-server then context-shifts and
    returns HTTP 200 with ``prompt_eval_count`` pinned near ``num_ctx``. Clients
    had to infer overflow from that pin (and sometimes ``done_reason: length``).
    We make the signal explicit when we know the original admit token count.

    Detection: ``original_prompt_tokens > num_ctx``, or
    ``prompt_eval_count`` within 8 of ``num_ctx`` while original > evaluated.
    ``done_reason: length`` is a common companion signal but is not required —
    length can also mean ``num_predict`` exhausted on a fully fit prompt.
    """
    if num_ctx is None or num_ctx <= 0:
        return {}
    pec = metrics.get("prompt_eval_count", 0)
    if pec <= 0:
        return {}
    orig = original_prompt_tokens or 0
    # Why clear orig when it already fits: a prompt that fills the window is not
    # an overflow; pinning near num_ctx alone is expected for long-but-valid inputs.
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
