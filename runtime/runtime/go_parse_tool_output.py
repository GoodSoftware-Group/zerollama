"""Parse tool output via Go (builtin parsers + template tools.Parser)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from runtime.go_render_chat import _go_base_url, _go_render_enabled


def _post_json(path: str, body: dict[str, Any], *, timeout_s: float = 2.0) -> dict[str, Any] | None:
    url = f"{_go_base_url()}{path}"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            out = json.loads(resp.read().decode())
        return out if isinstance(out, dict) else None
    except (OSError, urllib.error.URLError, ValueError, TypeError, json.JSONDecodeError):
        return None


def go_stream_parse_available(meta: dict[str, Any] | None) -> bool:
    """True when a builtin family parser is required (not generic JSON tag parsing)."""
    if not _go_render_enabled() or not meta:
        return False
    if meta.get("has_tool_support"):
        return True
    parser = meta.get("parser")
    return isinstance(parser, str) and bool(parser.strip())


def parse_tool_output_via_go(
    model: str,
    content: str,
    tools: list[dict[str, Any]] | None,
    *,
    messages: list[dict[str, Any]] | None = None,
    done: bool = True,
    think: Any = None,
    timeout_s: float = 2.0,
) -> dict[str, Any] | None:
    if not _go_render_enabled() or not str(model).strip():
        return None
    body: dict[str, Any] = {
        "model": model,
        "content": content,
        "done": done,
    }
    if tools:
        body["tools"] = tools
    if messages:
        body["messages"] = messages
    if think is not None:
        body["think"] = think
    return _post_json("/internal/parse-tool-output", body, timeout_s=timeout_s)


class GoToolParseStreamSession:
    """Stateful streaming parser session (Q4 — matches ggml ChatHandler)."""

    def __init__(self, session_id: str, method: str = "") -> None:
        self.session_id = session_id
        self.method = method

    @classmethod
    def open(
        cls,
        model: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]] | None = None,
        *,
        think: Any = None,
        timeout_s: float = 2.0,
    ) -> GoToolParseStreamSession | None:
        if not _go_render_enabled() or not str(model).strip():
            return None
        body: dict[str, Any] = {"model": model, "tools": tools}
        if messages:
            body["messages"] = messages
        if think is not None:
            body["think"] = think
        out = _post_json(
            "/internal/parse-tool-output/session", body, timeout_s=timeout_s
        )
        if not out or not out.get("session_id"):
            return None
        return cls(str(out["session_id"]), str(out.get("method") or ""))

    def add(
        self, content: str, *, done: bool = False, timeout_s: float = 2.0
    ) -> dict[str, Any] | None:
        return _post_json(
            "/internal/parse-tool-output/chunk",
            {
                "session_id": self.session_id,
                "content": content,
                "done": done,
            },
            timeout_s=timeout_s,
        )

    def close(self, timeout_s: float = 1.0) -> None:
        _post_json(
            "/internal/parse-tool-output/close",
            {"session_id": self.session_id},
            timeout_s=timeout_s,
        )
