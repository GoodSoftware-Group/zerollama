"""Sampler option parsing and llama-server payload merge (Phase 14)."""

from __future__ import annotations

from runtime.worker.sampler_options import (
    SamplerOptions,
    apply_sampler_to_completion_payload,
    sampler_options_from_dict,
    sampler_to_llama_cpp_kwargs,
)


def test_sampler_options_none_without_keys():
    assert sampler_options_from_dict({}) is None
    assert sampler_options_from_dict({"num_ctx": 4096}) is None
    assert sampler_options_from_dict(None) is None


def test_sampler_options_temperature_zero_is_greedy():
    opts = sampler_options_from_dict({"temperature": 0})
    assert opts is not None
    assert opts.greedy_only is True


def test_sampler_options_parses_temperature():
    opts = sampler_options_from_dict({"temperature": 0.2, "top_k": 20})
    assert opts is not None
    assert not opts.greedy_only
    assert opts.temperature == 0.2
    assert opts.top_k == 20
    assert opts.top_p == 0.9  # Ollama default when omitted


def test_apply_sampler_to_completion_payload():
    payload: dict = {"prompt": "hi", "stream": False}
    apply_sampler_to_completion_payload(
        payload,
        SamplerOptions(temperature=0.5, top_k=10, top_p=0.95),
    )
    assert payload["temperature"] == 0.5
    assert payload["top_k"] == 10
    assert payload["top_p"] == 0.95


def test_sampler_to_llama_cpp_kwargs_greedy():
    kw = sampler_to_llama_cpp_kwargs(None)
    assert kw["temperature"] == 0.0


def test_sampler_to_llama_cpp_kwargs_stochastic():
    kw = sampler_to_llama_cpp_kwargs(
        SamplerOptions(temperature=0.5, top_k=10, seed=7)
    )
    assert kw["temperature"] == 0.5
    assert kw["top_k"] == 10
    assert kw["seed"] == 7


def test_apply_sampler_greedy_sends_temperature_zero():
    payload: dict = {"prompt": "hi"}
    apply_sampler_to_completion_payload(payload, SamplerOptions(greedy_only=True))
    assert payload["temperature"] == 0.0
    assert "top_k" not in payload
