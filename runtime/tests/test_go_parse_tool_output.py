from unittest.mock import patch

from runtime.server.chat_tools import parse_completion_tool_calls


def test_parse_completion_prefers_go():
    tools = [{"type": "function", "function": {"name": "f"}}]
    with patch(
        "runtime.go_parse_tool_output.parse_tool_output_via_go",
        return_value={
            "content": "ok",
            "method": "harmony",
            "tool_calls": [
                {
                    "id": "call_abc",
                    "function": {"name": "f", "arguments": {"x": 1}},
                }
            ],
        },
    ):
        calls, content = parse_completion_tool_calls(
            "raw",
            tools,
            model="m",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert content == "ok"
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "f"
