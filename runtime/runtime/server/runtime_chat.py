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
    from runtime.invariants import think_forces_legacy

    if logprobs or think_forces_legacy(think):
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
    """Legacy role-line prompt when Go Modelfile render is unavailable.

    Do not use this for models that ship a real chat template (ChatML, Llama,
    …). Spoken duplex and /api/chat must call :func:`resolve_chat_prompt` so
    the model does not continue ``User:`` / ``Assistant:`` stage directions.
    """
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


def resolve_chat_prompt(
    model: str,
    messages: list[dict[str, Any]],
    *,
    think: Any = None,
    num_ctx: int | None = None,
    n_predict: int | None = None,
) -> str:
    """Modelfile / builtin template via Go ``/internal/render-chat``, else legacy lines.

    Uses ``truncate=false`` so Go skips Python vocab tokenize. Why: on this host
    in-process libllama SEGV's the runtime; token-accurate truncate is unsafe
    until that path is fixed. Duplex history is short; ChatML matters more than
    prefix drop. Legacy ``User:`` / ``Assistant:`` makes the ear speak role labels.
    """
    from runtime.go_render_chat import render_chat_via_go

    rendered = render_chat_via_go(
        model,
        messages,
        think=think,
        num_ctx=num_ctx,
        num_predict=n_predict,
        truncate=False,
    )
    if rendered:
        prompt = rendered.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            return prompt
    return messages_to_prompt(messages)
