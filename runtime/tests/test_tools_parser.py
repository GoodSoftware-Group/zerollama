import json

from runtime.server.chat_tools import (
    messages_to_tools_prompt,
    parse_completion_tool_calls,
)
from runtime.server.tools_parser import ToolCallParser


def test_parser_json_tool_call():
    tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
        }
    ]
    text = 'Sure. {"name": "get_weather", "arguments": {"city": "Paris"}}'
    calls, content = parse_completion_tool_calls(text, tools)
    assert content.strip() == "Sure."
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert calls[0]["function"]["arguments"]["city"] == "Paris"
    assert calls[0].get("id", "").startswith("call_")


def test_parser_streaming_add():
    tools = [
        {"type": "function", "function": {"name": "add", "parameters": {}}},
    ]
    p = ToolCallParser(tag="{", tools=tools)
    c1, t1 = p.add('{"name": "add", "arguments": {"a": 1')
    assert c1 == [] and t1 == ""
    c2, t2 = p.add(', "b": 2}}')
    assert len(c2) == 1
    assert c2[0]["function"]["name"] == "add"


def test_messages_to_tools_prompt_includes_schema():
    prompt = messages_to_tools_prompt(
        [{"role": "user", "content": "weather?"}],
        [{"type": "function", "function": {"name": "get_weather"}}],
    )
    assert "get_weather" in prompt
    assert "User: weather?" in prompt
    assert prompt.strip().endswith("Assistant:")
