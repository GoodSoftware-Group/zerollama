"""Render chat prompts via Go Modelfile templates (Phase 12 Q3)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from runtime.go_internal_url import connectable_go_base_url as _go_base_url


def _go_render_enabled() -> bool:
    raw = os.environ.get("ZEROLLAMA_RUNTIME_GO_RENDER_CHAT", "auto").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    return True


def render_chat_via_go(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    think: Any = None,
    num_ctx: int | None = None,
    num_predict: int | None = None,
    truncate: bool = True,
    timeout_s: float = 2.0,
) -> dict[str, Any] | None:
    """POST /internal/render-chat on the Go daemon; None when unavailable."""
    if not _go_render_enabled() or not str(model).strip():
        return None
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "truncate": truncate,
    }
    if tools:
        body["tools"] = tools
    if think is not None:
        body["think"] = think
    if num_ctx is not None and num_ctx > 0:
        body["num_ctx"] = num_ctx
    if num_predict is not None and num_predict > 0:
        body["num_predict"] = num_predict
    url = f"{_go_base_url()}/internal/render-chat"
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
        if isinstance(out, dict) and out.get("prompt"):
            return out
    except (OSError, urllib.error.URLError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return None
