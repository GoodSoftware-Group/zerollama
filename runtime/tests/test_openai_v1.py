from runtime.server.openai_v1 import (
    completion_json,
    v1_max_tokens,
    v1_needs_legacy,
)


def test_v1_needs_legacy_tools():
    assert not v1_needs_legacy({"messages": [], "tools": [{"type": "function"}]})


def test_v1_needs_legacy_image():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "http://x"}}],
            }
        ]
    }
    assert v1_needs_legacy(body)


def test_v1_text_ok():
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert not v1_needs_legacy(body)


def test_v1_think_false_uses_runtime():
    assert not v1_needs_legacy({"messages": [], "think": False})


def test_v1_max_tokens_accepts_float():
    assert v1_max_tokens({"max_tokens": 64.0}) == 64


def test_v1_max_tokens_from_options():
    assert v1_max_tokens({"options": {"num_predict": 32}}) == 32


def test_v1_needs_legacy_reasoning_effort():
    assert v1_needs_legacy({"messages": [], "reasoning_effort": "high"})


def test_v1_needs_legacy_message_reasoning():
    body = {
        "messages": [
            {"role": "assistant", "content": "", "reasoning": "thoughts"},
        ]
    }
    assert v1_needs_legacy(body)


def test_completion_json_shape():
    out = completion_json("m", "hello")
    assert out["object"] == "chat.completion"
    assert out["choices"][0]["message"]["content"] == "hello"
