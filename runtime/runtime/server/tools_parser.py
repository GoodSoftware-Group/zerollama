"""Stream parser for tool calls in model output (parity with Go tools.Parser)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class _State(Enum):
    LOOKING = auto()
    CALLING = auto()
    DONE = auto()


@dataclass
class ToolCallParser:
    """Parse tool calls from completion text using a delimiter tag (default ``{``)."""

    tag: str
    tools: list[dict[str, Any]]
    state: _State = _State.LOOKING
    buffer: bytearray = field(default_factory=bytearray)
    n: int = 0

    def add(self, text: str) -> tuple[list[dict[str, Any]], str]:
        if self.state == _State.DONE:
            return [], text
        if text:
            self.buffer.extend(text.encode("utf-8"))
        return self._drain()

    def _drain(self) -> tuple[list[dict[str, Any]], str]:
        calls: list[dict[str, Any]] = []
        content = ""

        if self.state == _State.LOOKING:
            i, found = self._find_tag()
            if i == -1:
                content = self.buffer.decode("utf-8", errors="replace")
                self.buffer.clear()
            else:
                content = self.buffer[:i].decode("utf-8", errors="replace")
                self.buffer = self.buffer[i:]
            if self.tag in ("{", "[") and content.strip():
                combined = content + self.buffer.decode("utf-8", errors="replace")
                idx = combined.find(self.tag)
                if idx > 0:
                    content = combined[:idx].strip()
                    self.buffer = bytearray(combined[idx:].encode("utf-8"))
                    self.state = _State.CALLING
                else:
                    self.state = _State.DONE
                    self.buffer.clear()
                    return [], combined
            if not found:
                return [], content
            self.state = _State.CALLING

        while True:
            call = self._parse_tool_call()
            if call is None:
                break
            calls.append(call)

        if self._done():
            self.state = _State.DONE
            tail = self.buffer.decode("utf-8", errors="replace")
            self.buffer.clear()
            content = (content + tail).strip()
        return calls, content

    def _find_tag(self) -> tuple[int, bool]:
        tag_b = self.tag.encode("utf-8")
        buf = bytes(self.buffer)
        i = buf.find(tag_b)
        if i > -1:
            return i, True
        max_len = min(len(buf), len(tag_b))
        for n in range(max_len, 0, -1):
            if buf.endswith(tag_b[:n]):
                return len(buf) - n, False
        return -1, False

    def _parse_tool_call(self) -> dict[str, Any] | None:
        tool, end = _find_tool(self.tools, bytes(self.buffer))
        if tool is None:
            return None
        args, arg_end = _find_arguments(tool, bytes(self.buffer))
        if args is None:
            return None
        if arg_end > end:
            end = arg_end
        name = _tool_name(tool)
        tc = {
            "function": {
                "name": name,
                "arguments": args,
                "index": self.n,
            }
        }
        self.n += 1
        self.buffer = self.buffer[end:]
        return tc

    def _done(self) -> bool:
        if self.tag == "{":
            open_c, close_c = ord("{"), ord("}")
        elif self.tag == "[":
            open_c, close_c = ord("["), ord("]")
        else:
            return False
        count = 0
        for b in self.buffer:
            if b == open_c:
                count += 1
            elif b == close_c:
                count -= 1
                if count == 0:
                    return True
        return False


def _tool_name(tool: dict[str, Any]) -> str:
    fn = tool.get("function")
    if isinstance(fn, dict):
        return str(fn.get("name", ""))
    return ""


def _find_tool(tools: list[dict[str, Any]], buf: bytes) -> tuple[dict[str, Any] | None, int]:
    if not buf:
        return None, 0
    names = [_tool_name(t) for t in tools if _tool_name(t)]
    longest = max((len(n) for n in names), default=0)
    for i in range(1, min(len(buf), longest) + 1):
        tail = buf[-i:]
        for name in names:
            nb = name.encode("utf-8")
            if len(tail) < len(nb) and nb.startswith(tail):
                return None, 0
    found: dict[str, Any] | None = None
    start = -1
    end = 0
    for tool in tools:
        name = _tool_name(tool)
        if not name:
            continue
        nb = name.encode("utf-8")
        pos = buf.find(nb)
        if pos == -1:
            continue
        if start != -1:
            if pos > start:
                continue
            if pos == start and len(name) <= len(_tool_name(found or {})):
                continue
        found = tool
        start = pos
        end = pos + len(nb)
    if found is not None:
        return found, end
    return None, 0


def _find_arguments(
    tool: dict[str, Any], buffer: bytes
) -> tuple[dict[str, Any] | None, int]:
    if not buffer:
        return None, 0
    name = _tool_name(tool)
    start = -1
    braces = 0
    in_string = False
    escaped = False
    for i, c in enumerate(buffer):
        if escaped:
            escaped = False
            continue
        if c == ord("\\"):
            escaped = True
            continue
        if c == ord('"'):
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == ord("{"):
            if braces == 0:
                start = i
            braces += 1
        elif c == ord("}"):
            braces -= 1
            if braces == 0 and start != -1:
                chunk = buffer[start : i + 1]
                try:
                    data = json.loads(chunk)
                except json.JSONDecodeError:
                    start = -1
                    continue
                args = _extract_arguments(data, name)
                if args is not None:
                    return args, i
                return data if isinstance(data, dict) else {}, i
            if braces < 0:
                braces = 0
    return None, 0


def _extract_arguments(data: Any, tool_name: str) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None

    def from_obj(obj: dict[str, Any]) -> dict[str, Any] | None:
        if "name" in obj:
            for key in ("arguments", "parameters"):
                val = obj.get(key)
                if isinstance(val, dict):
                    return val
                if isinstance(val, str):
                    try:
                        parsed = json.loads(val)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        pass
            return {}
        if tool_name in obj and isinstance(obj[tool_name], dict):
            return obj[tool_name]
        for v in obj.values():
            if isinstance(v, dict):
                got = from_obj(v)
                if got is not None:
                    return got
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        got = from_obj(item)
                        if got is not None:
                            return got
        return None

    return from_obj(data)
