"""Chat prompt helpers for runtime HTTP."""

from __future__ import annotations

from typing import Any


_LEGACY_CONTENT_TYPES = frozenset(
    {"video_url", "image_url", "input_image", "image", "video"}
)
_TEXT_CONTENT_TYPES = frozenset({"text", "input_text", ""})


def message_content_text(content: Any) -> str:
    """Plain text from Ollama string or OpenAI-style multipart content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if not isinstance(part, dict):
                continue
            ptype = str(part.get("type", "")).lower()
            if ptype in _TEXT_CONTENT_TYPES:
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def chat_needs_legacy(
    messages: list[dict[str, Any]] | None = None,
    *,
    tools: Any = None,
    logprobs: bool | None = False,
    think: Any = None,
) -> bool:
    """Vision/logprobs/think stay on ggml; plain tools use runtime chat_tools path."""
    del tools  # runtime supports tools via generic JSON prompt + parser
    if logprobs or think is not None:
        return True
    for m in messages or []:
        if m.get("images") or m.get("videos"):
            return True
        if m.get("thinking"):
            return True
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = str(part.get("type", "")).lower()
            if ptype in _LEGACY_CONTENT_TYPES:
                return True
            if ptype not in _TEXT_CONTENT_TYPES:
                return True
    return False


def messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for m in messages:
        role = str(m.get("role", "")).lower()
        content = message_content_text(m.get("content"))
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(content)
    return "\n".join(parts).strip()
