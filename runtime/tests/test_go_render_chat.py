from unittest.mock import patch

from runtime.server.chat_tools import resolve_tools_chat_prompt


def test_resolve_tools_chat_prompt_uses_go_render():
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "f"}}]

    with patch(
        "runtime.go_render_chat.render_chat_via_go",
        return_value={
            "prompt": "FROM_GO",
            "tool_tag": "<tool_call>",
            "has_tool_support": True,
        },
    ):
        prompt, tag, meta = resolve_tools_chat_prompt("llama3", messages, tools)
    assert prompt == "FROM_GO"
    assert tag == "<tool_call>"
    assert meta.get("has_tool_support") is True


def test_resolve_tools_requires_go_stream_for_harmony_parser():
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "f"}}]
    with patch(
        "runtime.go_render_chat.render_chat_via_go",
        return_value={
            "prompt": "FROM_GO",
            "tool_tag": "{",
            "parser": "harmony",
            "has_tool_support": True,
        },
    ):
        _prompt, _tag, meta = resolve_tools_chat_prompt("gpt-oss", messages, tools)
    assert meta.get("parser") == "harmony"
    assert meta.get("requires_go_tool_parser") is True


def test_resolve_tools_chat_prompt_propagates_truncated():
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "f"}}]
    with patch(
        "runtime.go_render_chat.render_chat_via_go",
        return_value={
            "prompt": "FROM_GO",
            "tool_tag": "{",
            "truncated": True,
            "truncate_mode": "heuristic",
        },
    ):
        _prompt, _tag, meta = resolve_tools_chat_prompt("llama3", messages, tools)
    assert meta.get("truncated") is True
    assert meta.get("truncate_mode") == "heuristic"


def test_resolve_tools_chat_prompt_propagates_truncate_mode():
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "f"}}]
    with patch(
        "runtime.go_render_chat.render_chat_via_go",
        return_value={
            "prompt": "FROM_GO",
            "tool_tag": "{",
            "truncate_mode": "tokenize",
        },
    ):
        _prompt, _tag, meta = resolve_tools_chat_prompt("llama3", messages, tools)
    assert meta.get("truncate_mode") == "tokenize"


def test_resolve_tools_chat_prompt_fallback():
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "f"}}]
    with patch("runtime.go_render_chat.render_chat_via_go", return_value=None):
        prompt, tag, _meta = resolve_tools_chat_prompt("m", messages, tools)
    assert "get_weather" not in prompt or "f" in prompt
    assert tag == "{"
