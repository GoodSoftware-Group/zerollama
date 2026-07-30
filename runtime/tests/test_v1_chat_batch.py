"""Unit tests for public batch chat validation (no GPU)."""

from types import SimpleNamespace

import pytest

from runtime.server.openai_v1 import run_v1_chat_completions_batch


class _FakeEngine:
    def __init__(self):
        self.config = SimpleNamespace(llama_parallel_slots=4)
        self.calls = []

    def generate_batch(self, prompts, n_predict=64, max_admit=4, *, options=None):
        self.calls.append(
            {
                "prompts": prompts,
                "n_predict": n_predict,
                "max_admit": max_admit,
                "options": options,
            }
        )
        return [
            SimpleNamespace(content=f"out-{i}", vram_num_ctx=None)
            for i in range(len(prompts))
        ]


def test_batch_same_model_and_rejects_tools():
    eng = _FakeEngine()
    out = run_v1_chat_completions_batch(
        eng,
        {
            "model": "m",
            "requests": [
                {"model": "m", "messages": [{"role": "user", "content": "a"}]},
                {"messages": [{"role": "user", "content": "b"}], "max_tokens": 16},
            ],
        },
    )
    assert out["count"] == 2
    assert out["completions"][0]["choices"][0]["message"]["content"] == "out-0"
    assert len(eng.calls[0]["prompts"]) == 2

    with pytest.raises(ValueError, match="same model"):
        run_v1_chat_completions_batch(
            eng,
            {
                "requests": [
                    {"model": "a", "messages": [{"role": "user", "content": "x"}]},
                    {"model": "b", "messages": [{"role": "user", "content": "y"}]},
                ]
            },
        )

    with pytest.raises(ValueError, match="tools"):
        run_v1_chat_completions_batch(
            eng,
            {
                "model": "m",
                "requests": [
                    {
                        "messages": [{"role": "user", "content": "x"}],
                        "tools": [{"type": "function", "function": {"name": "f"}}],
                    }
                ],
            },
        )


def test_batch_cap():
    eng = _FakeEngine()
    eng.config.llama_parallel_slots = 2
    reqs = [
        {"model": "m", "messages": [{"role": "user", "content": str(i)}]}
        for i in range(3)
    ]
    with pytest.raises(ValueError, match="exceeds max 2"):
        run_v1_chat_completions_batch(eng, {"requests": reqs})
