"""Tests for llama-server timings → Ollama metrics mapping."""

from runtime.llama_timings import metrics_from_llama_chunk


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


def test_metrics_from_llama_chunk_no_timings():
    assert metrics_from_llama_chunk({}) == {}
