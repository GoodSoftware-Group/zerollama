"""OpenAI /v1/chat/completions adapter for the Python runtime."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from runtime.server.chat_tools import (
    ToolParseUnavailableError,
    normalize_tools,
    parse_completion_tool_calls,
    resolve_tools_chat_prompt,
    stream_tool_chat_chunks,
)
from runtime.server.gguf_path import pop_gguf_path
from runtime.server.runtime_chat import chat_needs_legacy, messages_to_prompt


def _positive_int(value: Any) -> int | None:
    """Parse JSON numbers (int/float) for max_tokens / num_predict."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value > 0:
        return int(value)
    return None


def v1_max_tokens(body: dict[str, Any]) -> int:
    """Completion cap from OpenAI max_tokens, else options.num_predict, else 128."""
    n = _positive_int(body.get("max_tokens"))
    if n is not None:
        return n
    opts = v1_request_options(body)
    n = _positive_int(opts.get("num_predict"))
    if n is not None:
        return n
    return 128


def _v1_think_needs_legacy(think: Any) -> bool:
    """think:false disables legacy; other think values need ggml."""
    if think is None:
        return False
    if think is False:
        return False
    if isinstance(think, bool):
        return think
    return True


def v1_needs_legacy(body: dict[str, Any]) -> bool:
    """Vision/logprobs/think/reasoning stay on ggml; plain tools use runtime."""
    if _v1_think_needs_legacy(body.get("think")):
        return True
    reff = body.get("reasoning_effort")
    if isinstance(reff, str) and reff.strip():
        return True
    if body.get("reasoning") is not None:
        return True
    for m in body.get("messages") or []:
        if not isinstance(m, dict):
            continue
        if m.get("reasoning") or m.get("thinking"):
            return True
    think_val = body.get("think")
    legacy_think = None if think_val is False else think_val
    return chat_needs_legacy(
        body.get("messages") or [],
        tools=body.get("tools"),
        logprobs=bool(body.get("logprobs")),
        think=legacy_think,
    )


def v1_request_options(body: dict[str, Any]) -> dict[str, Any]:
    """Merge Ollama-shaped inference options from a v1 JSON body.

    Why: Go ``runtimeV1ProxyOptions`` injects ``options.gguf`` from the manifest; direct
    :8081 callers can pass ``options.gguf`` / ``options.num_ctx`` (or top-level ``num_ctx``).
    """
    opts: dict[str, Any] = {}
    raw = body.get("options")
    if isinstance(raw, dict):
        opts.update(raw)
    extra = body.get("extra_body")
    if isinstance(extra, dict):
        ex_opts = extra.get("options")
        if isinstance(ex_opts, dict):
            opts.update(ex_opts)
    for key in ("num_ctx", "num_predict"):
        if key in body and key not in opts:
            opts[key] = body[key]
    mt = _positive_int(body.get("max_tokens"))
    if mt is not None and "num_predict" not in opts:
        opts["num_predict"] = mt
    return opts


@dataclass
class V1ChatPrepared:
    model: str
    messages: list[dict[str, Any]]
    prompt: str
    tools: list[dict[str, Any]]
    tool_tag: str
    tools_meta: dict[str, Any]
    n_predict: int
    gguf: Path | None
    options: dict[str, Any]
    num_ctx: int | None


def prepare_v1_chat(engine: Any, body: dict[str, Any]) -> V1ChatPrepared:
    """Build prompt + Phase 13 ``num_ctx`` (resolve + optional clamp) for v1 chat."""
    from runtime.format_constraint import merge_format_into_options, parse_format_constraint

    model = str(body.get("model", ""))
    messages = list(body.get("messages") or [])
    opts = v1_request_options(body)
    # M15f: fold response_format / format into options for llama-server constraints.
    opts = merge_format_into_options(
        opts,
        format_value=body.get("format"),
        response_format=body.get("response_format"),
    )
    tools = normalize_tools(body.get("tools"))
    if tools and parse_format_constraint(
        body.get("format"), response_format=body.get("response_format")
    ):
        raise ValueError("grammar is not supported together with tools")
    gguf = pop_gguf_path(opts)
    num_ctx, _meta = engine.resolve_num_ctx_for_request(gguf, options=opts)
    n_predict = v1_max_tokens(body)
    tool_tag = "{"
    tools_meta: dict[str, Any] = {}
    if tools:
        prompt, tool_tag, tools_meta = resolve_tools_chat_prompt(
            model,
            messages,
            tools,
            think=body.get("think"),
            num_ctx=num_ctx,
            n_predict=n_predict,
        )
    else:
        prompt = messages_to_prompt(messages)
    return V1ChatPrepared(
        model=model,
        messages=messages,
        prompt=prompt,
        tools=tools,
        tool_tag=tool_tag,
        tools_meta=tools_meta,
        n_predict=n_predict,
        gguf=gguf,
        options=opts,
        num_ctx=num_ctx,
    )


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def completion_json(
    model: str,
    content: str,
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    vram_num_ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    finish = "stop"
    if tool_calls:
        message["tool_calls"] = _openai_tool_calls(tool_calls)
        finish = "tool_calls"
    out: dict[str, Any] = {
        "id": _completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": "fp_zerollama_runtime",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish,
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
    # Ollama-shaped extension on OpenAI responses — why: clamp visibility without breaking clients.
    if vram_num_ctx:
        out["vram_num_ctx"] = vram_num_ctx
    return out


def _openai_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, tc in enumerate(calls):
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name", ""))
        args = fn.get("arguments", {})
        if not isinstance(args, dict):
            args = {}
        out.append(
            {
                "id": tc.get("id") or f"call_{i}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args),
                },
            }
        )
    return out


def stream_openai_sse(
    engine: Any,
    body: dict[str, Any],
    *,
    prefill_cancel: Any | None = None,
) -> Iterator[str]:
    prep = prepare_v1_chat(engine, body)
    cid = _completion_id()
    created = int(time.time())

    def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
        choice: dict[str, Any] = {"index": 0, "delta": delta}
        if finish is not None:
            choice["finish_reason"] = finish
        else:
            choice["finish_reason"] = None
        payload = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": prep.model,
            "system_fingerprint": "fp_zerollama_runtime",
            "choices": [choice],
        }
        return f"data: {json.dumps(payload)}\n\n"

    yield chunk({"role": "assistant"})
    try:
        if prep.tools:
            parts = stream_tool_chat_chunks(
                engine,
                prep.prompt,
                prep.model,
                prep.tools,
                n_predict=prep.n_predict,
                gguf=prep.gguf,
                num_ctx=prep.num_ctx,
                options=prep.options,
                tag=prep.tool_tag,
                messages=prep.messages,
                think=body.get("think"),
                tools_meta=prep.tools_meta,
                prefill_cancel=prefill_cancel,
            )
        else:
            parts = engine.stream_chat(
                prep.prompt,
                prep.model,
                n_predict=prep.n_predict,
                gguf=prep.gguf,
                num_ctx=prep.num_ctx,
                options=prep.options,
                prefill_cancel=prefill_cancel,
            )

        for part in parts:
            msg = part.get("message") or {}
            content = msg.get("content", "")
            done = bool(part.get("done"))
            delta: dict[str, Any] = {}
            if content:
                delta["content"] = content
            tcalls = msg.get("tool_calls")
            if tcalls:
                delta["tool_calls"] = _openai_tool_calls(tcalls)
            if delta:
                yield chunk(delta)
            if done:
                reason = part.get("done_reason") or "stop"
                if reason == "tool_calls":
                    reason = "tool_calls"
                # WHY preserve cancelled: HTTP disconnect prefill abort is not a normal
                # stop — agents use finish_reason to retry or adjust context.
                elif reason not in ("stop", "length", "cancelled"):
                    reason = "stop"
                yield chunk({}, finish=reason)
                break
    except ToolParseUnavailableError as e:
        err = {"error": {"message": str(e), "type": "tool_parse_unavailable"}}
        yield f"data: {json.dumps(err)}\n\n"
    except Exception as e:
        # Why: load/generate errors after the first SSE chunk left connections open
        # without [DONE], so Go/curl proxies hung until client timeout.
        err = {"error": {"message": str(e), "type": type(e).__name__}}
        yield f"data: {json.dumps(err)}\n\n"
    yield "data: [DONE]\n\n"


def run_v1_chat_completion(
    engine: Any,
    body: dict[str, Any],
    *,
    prefill_cancel: Any | None = None,
) -> dict[str, Any]:
    """Non-stream v1 chat completion (tools + Phase 13 ctx)."""
    prep = prepare_v1_chat(engine, body)
    if not prep.prompt:
        raise ValueError("empty messages")
    result = engine.generate(
        prep.prompt,
        n_predict=prep.n_predict,
        gguf=prep.gguf,
        num_ctx=prep.num_ctx,
        options=prep.options,
        prefill_cancel=prefill_cancel,
    )
    if prep.tools:
        calls, content = parse_completion_tool_calls(
            result.content,
            prep.tools,
            tag=prep.tool_tag,
            model=prep.model,
            messages=prep.messages,
            think=body.get("think"),
            tools_meta=prep.tools_meta,
        )
        return completion_json(
            prep.model,
            content,
            tool_calls=calls or None,
            vram_num_ctx=result.vram_num_ctx,
        )
    return completion_json(
        prep.model, result.content, vram_num_ctx=result.vram_num_ctx
    )


def _batch_item_needs_reject(body: dict[str, Any]) -> str | None:
    """Batch path rejects tools/vision/think (same floor as single-request runtime)."""
    if body.get("tools"):
        return "batch chat does not support tools"
    if body.get("tool_choice") not in (None, "none"):
        return "batch chat does not support tool_choice"
    if v1_needs_legacy(body):
        return "batch chat does not support vision/logprobs/think (use single /v1/chat/completions)"
    return None


def run_v1_chat_completions_batch(
    engine: Any,
    body: dict[str, Any],
) -> dict[str, Any]:
    """Non-stream batch of independent chat bodies → ``generate_batch``.

    WHY public (vs /internal/generate-batch): Hermes wants OpenAI-shaped fan-out
    without N HTTP round-trips; Python already does decode batching. Cap N at
    ``min(8, llama_parallel_slots)``. Same-model + shared options.gguf required.
    Streaming deferred (NDJSON with seq_idx) to a later iteration.

    WHY wrapper response (not a bare list)::
        ``{object: "chat.completion.batch", model, count, completions}`` with
        ``completions[i]`` corresponding to ``requests[i]``. A bare
        ``list[chat.completion]`` collides with OpenAI async Batch API mental
        models and left OpenAPI saying only "OpenAI-shaped list / wrapper" —
        that underspec blocked Hermes aux-client grouping work.

    WHY reject mixed models here (not group server-side)::
        ``generate_batch`` is one decode fan-out on one loaded GGUF. Grouping
        across models is a client concern (one POST per model, chunk ≤ cap).
    """
    requests = body.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("requests must be a non-empty list")
    parallel = int(getattr(engine.config, "llama_parallel_slots", 1) or 1)
    cap = min(8, max(1, parallel))
    if len(requests) > cap:
        raise ValueError(f"batch size {len(requests)} exceeds max {cap}")

    model = ""
    prompts: list[str] = []
    n_predict = 128
    shared_opts: dict[str, Any] = {}
    for i, item in enumerate(requests):
        if not isinstance(item, dict):
            raise ValueError(f"requests[{i}] must be an object")
        if reason := _batch_item_needs_reject(item):
            raise ValueError(f"requests[{i}]: {reason}")
        item_model = str(item.get("model") or body.get("model") or "").strip()
        if not item_model:
            raise ValueError(f"requests[{i}]: model is required")
        if not model:
            model = item_model
        elif item_model != model:
            # WHY 400 not silent split: callers must own ordering/reassembly across
            # model groups; inventing a multi-runner batch here fights VRAM broker.
            raise ValueError("batch chat requires the same model for all requests")
        messages = list(item.get("messages") or [])
        prompt = messages_to_prompt(messages)
        if not prompt.strip():
            raise ValueError(f"requests[{i}]: empty messages")
        prompts.append(prompt)
        n_predict = max(n_predict, v1_max_tokens(item))
        opts = v1_request_options(item)
        if i == 0:
            shared_opts = opts
        else:
            # Merge conservatively: first request wins for gguf/num_ctx; later
            # rows may only add prompt_cache_keys list entries below.
            for k, v in opts.items():
                if k not in shared_opts:
                    shared_opts[k] = v

    # Per-row cache keys when present.
    keys: list[str | None] = []
    for item in requests:
        opts = v1_request_options(item) if isinstance(item, dict) else {}
        from runtime.cache_bridge import extract_prompt_cache_key

        keys.append(extract_prompt_cache_key(opts) or extract_prompt_cache_key(item))
    if any(k for k in keys):
        shared_opts["prompt_cache_keys"] = keys

    # Shared top-level options (Go injects gguf once).
    top_opts = v1_request_options(body)
    for k, v in top_opts.items():
        if k not in shared_opts or shared_opts.get(k) in (None, ""):
            shared_opts[k] = v

    results = engine.generate_batch(
        prompts,
        n_predict=n_predict,
        max_admit=min(cap, len(prompts)),
        options=shared_opts,
    )
    completions = [
        completion_json(model, r.content, vram_num_ctx=getattr(r, "vram_num_ctx", None))
        for r in results
    ]
    # Stable wire — keep keys explicit so OpenAPI / Hermes clients stay aligned.
    return {
        "object": "chat.completion.batch",
        "model": model,
        "completions": completions,
        "count": len(completions),
    }
