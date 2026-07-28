"""Tests for llama-server timings → Ollama metrics mapping."""

from runtime.llama_timings import detect_context_overflow, metrics_from_llama_chunk


def test_metrics_from_llama_chunk_cache_hit():
    out = metrics_from_llama_chunk(
        {
            "timings": {
                "cache_n": 120,
                "prompt_n": 8,
                "prompt_ms": 12.5,
                "predicted_n": 16,
                "predicted_ms": 40.0,
            }
        }
    )
    assert out["prompt_eval_cached_count"] == 120
    assert out["cached_prompt_tokens"] == 120
    assert out["prompt_eval_count"] == 128
    assert out["prompt_eval_duration"] == 12_500_000
    assert out["eval_count"] == 16
    assert out["eval_duration"] == 40_000_000


def test_metrics_from_llama_chunk_host_storage_tiers():
    out = metrics_from_llama_chunk(
        {
            "timings": {"cache_n": 50, "prompt_n": 5},
            "cached_tokens_host": 40,
            "cached_tokens_storage": 10,
            "cached_tokens_storage_backend": "redis",
        }
    )
    assert out["cached_prompt_tokens"] == 50
    assert out["cached_tokens_host"] == 40
    assert out["cached_tokens_storage"] == 10
    assert out["cached_tokens_storage_backend"] == "redis"


def test_metrics_from_llama_chunk_cache_creation():
    out = metrics_from_llama_chunk(
        {
            "timings": {"cache_n": 100, "prompt_n": 20},
            "cache_creation_tokens": 80,
        }
    )
    assert out["cached_prompt_tokens"] == 100
    assert out["cache_creation_tokens"] == 80
    assert out["created_cache_tokens"] == 80
    assert out["prompt_eval_cache_creation_count"] == 80


def test_merge_cache_tier_details_omits_zeros():
    from runtime.llama_timings import merge_cache_tier_details

    assert merge_cache_tier_details({}) == {}
    out = merge_cache_tier_details({}, host=12, storage=0)
    assert out == {"cached_tokens_host": 12, "prompt_eval_cached_host": 12}


def test_metrics_from_llama_chunk_no_timings():
    assert metrics_from_llama_chunk({}) == {}


def test_detect_context_overflow_triggered():
    """44k-token prompt truncated to fit num_ctx=8192."""
    result = detect_context_overflow(
        {"prompt_eval_count": 8184},
        num_ctx=8192,
        original_prompt_tokens=44000,
    )
    assert result["prompt_truncated"] is True
    assert result["original_prompt_tokens"] == 44000


def test_detect_context_overflow_no_overflow():
    """Small prompt fits in context — no truncation."""
    result = detect_context_overflow(
        {"prompt_eval_count": 500},
        num_ctx=8192,
        original_prompt_tokens=500,
    )
    assert result == {}


def test_detect_context_overflow_no_num_ctx():
    result = detect_context_overflow(
        {"prompt_eval_count": 8184},
        num_ctx=None,
        original_prompt_tokens=44000,
    )
    assert result == {}


def test_detect_context_overflow_pinned_at_window():
    """prompt_eval_count near num_ctx with original > actual."""
    result = detect_context_overflow(
        {"prompt_eval_count": 8190},
        num_ctx=8192,
        original_prompt_tokens=10000,
    )
    assert result["prompt_truncated"] is True
    assert result["original_prompt_tokens"] == 10000
