"""Tool-aware chat prompts and response parsing for the Python runtime."""

from __future__ import annotations

import json
import secrets
import string
from typing import Any, Iterator

from runtime.server.runtime_chat import message_content_text
from runtime.server.tools_parser import ToolCallParser

_ALNUM = string.ascii_letters + string.digits


class ToolParseUnavailableError(RuntimeError):
    """Builtin tool parser required but Go parse path is unavailable."""


def tool_call_id() -> str:
    return "call_" + "".join(secrets.choice(_ALNUM) for _ in range(8))


def normalize_tools(tools: list[Any] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            out.append(t)
    return out


def resolve_tools_chat_prompt(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    think: Any = None,
    num_ctx: int | None = None,
    n_predict: int | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Go Modelfile render when available; else generic JSON tools prompt."""
    from runtime.go_render_chat import render_chat_via_go

    meta: dict[str, Any] = {"has_tool_support": False, "parser": ""}
    rendered = render_chat_via_go(
        model, messages, tools, think=think, num_ctx=num_ctx, num_predict=n_predict
    )
    if rendered:
        tag = rendered.get("tool_tag")
        if not isinstance(tag, str) or not tag:
            tag = "{"
        meta["has_tool_support"] = bool(rendered.get("has_tool_support"))
        if isinstance(rendered.get("parser"), str):
            meta["parser"] = rendered["parser"]
        meta["requires_go_tool_parser"] = use_go_stream_tool_parser(meta)
        mode = rendered.get("truncate_mode")
        if isinstance(mode, str) and mode:
            meta["truncate_mode"] = mode
        if rendered.get("truncated") is True:
            meta["truncated"] = True
        return str(rendered["prompt"]), tag, meta
    return messages_to_tools_prompt(messages, tools), "{", meta


def use_go_stream_tool_parser(meta: dict[str, Any] | None) -> bool:
    from runtime.go_parse_tool_output import go_stream_parse_available

    return go_stream_parse_available(meta)


def _go_parse_usable(parsed: dict[str, Any]) -> bool:
    if parsed.get("method"):
        return True
    if parsed.get("tool_calls"):
        return True
    if parsed.get("thinking"):
        return True
    content = parsed.get("content")
    return isinstance(content, str) and bool(content)


def _yield_go_parse_deltas(
    model: str,
    created: str,
    parsed: dict[str, Any],
    sent_calls: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply one Go parse chunk; return (chunks to yield, updated sent_calls)."""
    out_chunks: list[dict[str, Any]] = []
    thinking = parsed.get("thinking")
    if isinstance(thinking, str) and thinking:
        msg = {"role": "assistant", "thinking": thinking}
        out_chunks.append(
            {
                "model": model,
                "created_at": created,
                "message": msg,
                "done": False,
            }
        )
    calls, content = _tool_calls_from_go(parsed)
    if calls:
        sent_calls = assign_tool_call_ids(sent_calls + calls)
    if content:
        out_chunks.append(
            _chat_chunk(
                model,
                created,
                content,
                tool_calls=sent_calls if sent_calls else None,
            )
        )
    elif calls:
        out_chunks.append(
            _chat_chunk(
                model,
                created,
                "",
                tool_calls=sent_calls,
            )
        )
    return out_chunks, sent_calls


def _tool_calls_from_go(
    parsed: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    calls_raw = parsed.get("tool_calls")
    calls: list[dict[str, Any]] = []
    if isinstance(calls_raw, list):
        for tc in calls_raw:
            if isinstance(tc, dict):
                calls.append(_normalize_go_tool_call(tc))
    content = parsed.get("content")
    if not isinstance(content, str):
        content = ""
    return assign_tool_call_ids(calls), content


def _normalize_go_tool_call(tc: dict[str, Any]) -> dict[str, Any]:
    fn = tc.get("function")
    if not isinstance(fn, dict):
        return tc
    args = fn.get("arguments")
    if hasattr(args, "keys"):
        return tc
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    out = dict(tc)
    out_fn = dict(fn)
    out_fn["arguments"] = args
    out["function"] = out_fn
    return out


def _finish_go_raw_fallback(
    model: str,
    created: str,
    raw_buf: list[str],
    tools: list[dict[str, Any]],
    *,
    messages: list[dict[str, Any]] | None,
    think: Any,
    sent_calls: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from runtime.go_parse_tool_output import parse_tool_output_via_go

    text = "".join(raw_buf)
    if not text:
        return [], sent_calls
    parsed = parse_tool_output_via_go(
        model,
        text,
        tools,
        messages=messages,
        done=True,
        think=think,
    )
    if not parsed or not _go_parse_usable(parsed):
        raise ToolParseUnavailableError(
            f"Go tool parse failed after stream degradation for model {model!r}"
        )
    return _yield_go_parse_deltas(model, created, parsed, sent_calls)


def parse_completion_tool_calls(
    text: str,
    tools: list[dict[str, Any]],
    *,
    tag: str = "{",
    model: str = "",
    messages: list[dict[str, Any]] | None = None,
    think: Any = None,
    tools_meta: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    go_required = use_go_stream_tool_parser(tools_meta)
    if model:
        from runtime.go_parse_tool_output import parse_tool_output_via_go

        parsed = parse_tool_output_via_go(
            model,
            text,
            tools,
            messages=messages,
            done=True,
            think=think,
        )
        if parsed and _go_parse_usable(parsed):
            return _tool_calls_from_go(parsed)
        if go_required:
            raise ToolParseUnavailableError(
                f"Go tool parse unavailable for model {model!r}"
            )
    parser = ToolCallParser(tag=tag, tools=tools)
    calls, content = parser.add(text)
    extra, tail = parser.add("")
    calls.extend(extra)
    if tail and not content:
        content = tail
    return assign_tool_call_ids(calls), content


def messages_to_tools_prompt(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    tool_call_tag: str = "{",
) -> str:
    """Generic tools prompt (JSON tool calls; matches Go tools.Parser default tag)."""
    parts: list[str] = []
    system_lines: list[str] = []
    for m in messages:
        if str(m.get("role", "")).lower() == "system":
            system_lines.append(message_content_text(m.get("content")))
    if system_lines:
        parts.append("\n".join(system_lines).strip())
    if tools:
        parts.append(
            "You have access to the following tools (JSON schema):\n"
            + json.dumps(tools, indent=2)
        )
        parts.append(
            "To call a tool, output one or more JSON objects with "
            '"name" and "arguments" (or "parameters"). '
            "Do not wrap them in markdown fences."
        )
    for m in messages:
        role = str(m.get("role", "")).lower()
        if role == "system":
            continue
        if role == "tool":
            name = m.get("tool_name") or m.get("name") or "tool"
            parts.append(f"Tool result ({name}): {message_content_text(m.get('content'))}")
            continue
        if role == "assistant" and m.get("tool_calls"):
            calls = m.get("tool_calls")
            if isinstance(calls, list):
                for tc in calls:
                    if isinstance(tc, dict):
                        fn = tc.get("function") or tc
                        parts.append(
                            "Assistant tool call: "
                            + json.dumps(fn, ensure_ascii=False)
                        )
            content = message_content_text(m.get("content"))
            if content:
                parts.append(f"Assistant: {content}")
            continue
        content = message_content_text(m.get("content"))
        if role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        elif content:
            parts.append(content)
    parts.append("Assistant:")
    return "\n\n".join(p for p in parts if p).strip() + "\n"


def assign_tool_call_ids(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tc in calls:
        if not isinstance(tc, dict):
            continue
        entry = dict(tc)
        if not entry.get("id"):
            entry["id"] = tool_call_id()
        fn = entry.get("function")
        if isinstance(fn, dict) and "arguments" in fn:
            args = fn["arguments"]
            if not isinstance(args, dict):
                fn = dict(fn)
                fn["arguments"] = args if isinstance(args, dict) else {}
                entry["function"] = fn
        out.append(entry)
    return out


def stream_tool_chat_chunks(
    engine: Any,
    prompt: str,
    model: str,
    tools: list[dict[str, Any]],
    *,
    n_predict: int = 64,
    gguf: Any = None,
    num_ctx: int | None = None,
    options: dict | None = None,
    tag: str = "{",
    messages: list[dict[str, Any]] | None = None,
    think: Any = None,
    tools_meta: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield Ollama /api/chat NDJSON chunks with optional tool_calls."""
    meta = tools_meta or {}
    go_required = use_go_stream_tool_parser(meta)
    go_session = None
    parser = None if go_required else ToolCallParser(tag=tag, tools=tools)
    raw_buf: list[str] = []
    go_raw_fallback = False
    created = engine._utc_now()
    sent_calls: list[dict[str, Any]] = []

    if go_required and model:
        from runtime.go_parse_tool_output import GoToolParseStreamSession

        go_session = GoToolParseStreamSession.open(
            model, tools, messages, think=think
        )
        if go_session is None:
            raise ToolParseUnavailableError(
                f"Go tool parse session unavailable for model {model!r}"
            )

    # Forward vram_num_ctx from first stream_generate chunk — why: clamp meta must
    # reach /api/chat clients the same way as plain stream_generate.
    vram_api: dict[str, Any] | None = None
    try:
        for chunk in engine.stream_generate(
            prompt,
            model,
            n_predict,
            gguf=gguf,
            num_ctx=num_ctx,
            options=options,
        ):
            if vram_api is None and chunk.get("vram_num_ctx"):
                vram_api = chunk["vram_num_ctx"]
            piece = chunk.get("response") or ""
            done = bool(chunk.get("done"))

            if go_raw_fallback:
                if piece:
                    raw_buf.append(piece)
                if done:
                    break
                continue

            if go_session is not None and (piece or done):
                parsed = go_session.add(piece, done=done)
                if parsed is None:
                    go_session.close()
                    go_session = None
                    go_raw_fallback = True
                    if piece:
                        raw_buf.append(piece)
                    if done:
                        break
                    continue
                deltas, sent_calls = _yield_go_parse_deltas(
                    model, created, parsed, sent_calls
                )
                for out in deltas:
                    if vram_api and "vram_num_ctx" not in out:
                        out["vram_num_ctx"] = vram_api
                        vram_api = None
                    yield out
                if done:
                    break
                continue

            if piece and parser is not None:
                calls, content = parser.add(piece)
                if calls:
                    sent_calls = assign_tool_call_ids(sent_calls + calls)
                if content:
                    out = _chat_chunk(
                        model,
                        created,
                        content,
                        tool_calls=sent_calls if sent_calls else None,
                    )
                    if vram_api:
                        out["vram_num_ctx"] = vram_api
                        vram_api = None
                    yield out
            if done:
                break

        if go_raw_fallback:
            deltas, sent_calls = _finish_go_raw_fallback(
                model,
                created,
                raw_buf,
                tools,
                messages=messages,
                think=think,
                sent_calls=sent_calls,
            )
            for out in deltas:
                if vram_api and "vram_num_ctx" not in out:
                    out["vram_num_ctx"] = vram_api
                    vram_api = None
                yield out
        elif parser is not None:
            flush_calls, tail = parser.add("")
            if flush_calls:
                sent_calls = assign_tool_call_ids(sent_calls + flush_calls)
            if tail:
                tail_chunk = _chat_chunk(
                    model,
                    created,
                    tail,
                    tool_calls=sent_calls if sent_calls else None,
                )
                if vram_api:
                    tail_chunk["vram_num_ctx"] = vram_api
                    vram_api = None
                yield tail_chunk
    finally:
        if go_session is not None:
            go_session.close()

    reason = "tool_calls" if sent_calls else "stop"
    final = _chat_chunk(
        model,
        created,
        "",
        done=True,
        done_reason=reason,
        tool_calls=sent_calls if sent_calls else None,
    )
    if vram_api:
        final["vram_num_ctx"] = vram_api
    yield final


def _chat_chunk(
    model: str,
    created: str,
    content: str,
    *,
    done: bool = False,
    done_reason: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    out: dict[str, Any] = {
        "model": model,
        "created_at": created,
        "message": msg,
        "done": done,
    }
    if done_reason is not None:
        out["done_reason"] = done_reason
    return out
