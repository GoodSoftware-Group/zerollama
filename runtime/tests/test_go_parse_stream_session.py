from unittest.mock import patch

from runtime.go_parse_tool_output import GoToolParseStreamSession, go_stream_parse_available
from runtime.server.chat_tools import _yield_go_parse_deltas, use_go_stream_tool_parser


def test_go_stream_parse_available():
    assert go_stream_parse_available({"parser": "harmony"})
    assert go_stream_parse_available({"has_tool_support": True})
    assert not go_stream_parse_available({})


def test_use_go_stream_tool_parser():
    assert use_go_stream_tool_parser({"parser": "gemma4"})
    assert not use_go_stream_tool_parser({"template": True})


def test_go_tool_parse_stream_session():
    calls: list[dict] = []

    def fake_post(path, body, timeout_s=2.0):
        if "session_id" in body:
            calls.append(body)
            if body.get("done"):
                return {
                    "content": "",
                    "method": "harmony",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "f",
                                "arguments": {"x": 1},
                            }
                        }
                    ],
                }
            return {"content": "hi", "method": "harmony"}
        return {"session_id": "abc", "method": "harmony"}

    with (
        patch("runtime.go_parse_tool_output._go_render_enabled", return_value=True),
        patch("runtime.go_parse_tool_output._post_json", side_effect=fake_post),
    ):
        sess = GoToolParseStreamSession.open(
            "m",
            [{"type": "function", "function": {"name": "f"}}],
        )
        assert sess is not None
        mid = sess.add("hi", done=False)
        assert mid and mid.get("content") == "hi"
        fin = sess.add("", done=True)
        assert fin and fin.get("tool_calls")
        assert calls[-1]["done"] is True


def test_yield_go_parse_deltas_thinking():
    created = "2020-01-01T00:00:00Z"
    deltas, sent = _yield_go_parse_deltas(
        "m",
        created,
        {"thinking": "plan", "content": "ok"},
        [],
    )
    assert len(deltas) == 2
    assert deltas[0]["message"].get("thinking") == "plan"
    assert sent == []
