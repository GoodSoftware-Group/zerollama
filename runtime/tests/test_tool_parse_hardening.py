from unittest.mock import MagicMock, patch

import pytest

from runtime.server.chat_tools import (
    ToolParseUnavailableError,
    parse_completion_tool_calls,
    stream_tool_chat_chunks,
)


def test_parse_completion_raises_when_go_required_and_unavailable():
    tools = [{"type": "function", "function": {"name": "f"}}]
    with pytest.raises(ToolParseUnavailableError):
        parse_completion_tool_calls(
            "raw",
            tools,
            model="m",
            tools_meta={"parser": "harmony", "has_tool_support": True},
        )


def test_stream_raises_when_go_session_unavailable():
    eng = MagicMock()
    meta = {"parser": "harmony", "has_tool_support": True}
    with (
        patch("runtime.go_parse_tool_output._go_render_enabled", return_value=True),
        patch(
            "runtime.go_parse_tool_output.GoToolParseStreamSession.open",
            return_value=None,
        ),
    ):
        with pytest.raises(ToolParseUnavailableError):
            list(
                stream_tool_chat_chunks(
                    eng,
                    "prompt",
                    "m",
                    [{"type": "function", "function": {"name": "f"}}],
                    tools_meta=meta,
                )
            )


def test_stream_degrades_to_one_shot_parse():
    eng = MagicMock()
    meta = {"parser": "harmony"}
    chunks = [
        {"response": "part", "done": False},
        {"response": "", "done": True},
    ]
    eng.stream_generate.return_value = iter(chunks)
    session = MagicMock()
    session.add.side_effect = [None, None]
    session.close.return_value = None

    with (
        patch("runtime.go_parse_tool_output._go_render_enabled", return_value=True),
        patch(
            "runtime.go_parse_tool_output.GoToolParseStreamSession.open",
            return_value=session,
        ),
        patch(
            "runtime.go_parse_tool_output.parse_tool_output_via_go",
            return_value={
                "content": "ok",
                "method": "harmony",
                "tool_calls": [
                    {
                        "function": {
                            "name": "f",
                            "arguments": {"x": 1},
                        }
                    }
                ],
            },
        ),
    ):
        out = list(
            stream_tool_chat_chunks(
                eng,
                "prompt",
                "m",
                [{"type": "function", "function": {"name": "f"}}],
                tools_meta=meta,
            )
        )
    assert any(c.get("message", {}).get("content") == "ok" for c in out)
    assert any(c.get("done") for c in out)
