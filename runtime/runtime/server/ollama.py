"""Ollama-shaped HTTP models for /api/generate proxy compatibility."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class OllamaGenerateRequest(BaseModel):
    model: str = ""
    prompt: str
    stream: Optional[bool] = False
    options: dict[str, Any] = Field(default_factory=dict)
    # M15f: structured output / GBNF (forwarded from Go runtime proxy).
    format: Optional[Any] = None


class OllamaGenerateResponse(BaseModel):
    model: str = ""
    response: str = ""
    done: bool = True
    done_reason: Optional[str] = "stop"
    vram_num_ctx: Optional[dict[str, Any]] = None
    kv_decode_steps: Optional[int] = None
    # Why: Go proxy used to forward runtime /api/generate without truncation
    # metadata; clients only saw prompt_eval_count pinned at num_ctx.
    prompt_truncated: Optional[bool] = None
    original_prompt_tokens: Optional[int] = None


class OllamaChatRequest(BaseModel):
    model: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    stream: Optional[bool] = False
    options: dict[str, Any] = Field(default_factory=dict)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    logprobs: Optional[bool] = False
    think: Optional[Any] = None
    format: Optional[Any] = None
