"""HTTP sidecar (Phase 1–2).

Why FastAPI handlers use ``req: Model = Body()`` (no ``from __future__ import annotations``):
postponed annotations make FastAPI treat parameters as query fields (422 on ``query.req``).
Per-request weights: ``options.gguf`` (from Go manifest lookup) or ``LLAMA_MODEL`` env fallback.
"""

from typing import Any, Optional

from runtime.config import RuntimeConfig
from runtime.engine import InferenceEngine
from runtime.worker.llama_server import LlamaServerError


def create_app(
    config: Optional[RuntimeConfig] = None,
    engine: Optional[InferenceEngine] = None,
) -> Any:
    try:
        from fastapi import Body, FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as e:
        raise ImportError(
            "install serve extras: pip install -e '.[serve]'"
        ) from e

    from fastapi.responses import StreamingResponse

    from runtime.gpu_vram import resolve_num_ctx
    from runtime.server.gguf_path import pop_gguf_path
    from runtime.vram_env_apply import apply_exported_vram_env
    from runtime.vram_yaml_defaults import apply_vram_defaults_from_config

    # YAML vram: defaults first (16GB autoconfig); exported .env factor file is opt-in after.
    apply_vram_defaults_from_config()
    apply_exported_vram_env()
    from runtime.server.ollama import (
        OllamaChatRequest,
        OllamaGenerateRequest,
        OllamaGenerateResponse,
    )

    from runtime.server.access_log import log_request_in, log_response_out, runtime_queue_snapshot

    eng = engine or InferenceEngine(config)
    eng.maybe_auto_resume_inference()

    class CompletionRequest(BaseModel):
        prompt: str
        n_predict: int = Field(default=64, ge=1, le=8192)

    from starlette.requests import Request
    from starlette.responses import JSONResponse

    app = FastAPI(title="zerollama-runtime", version="0.2.0")

    from runtime.server.loopback import is_loopback_host

    @app.middleware("http")
    async def internal_loopback_only(request: Request, call_next):
        if request.url.path.startswith("/internal/"):
            client = request.client
            if client is None:
                return await call_next(request)
            if not is_loopback_host(client.host):
                return JSONResponse(
                    status_code=403,
                    content={"error": "internal endpoints are loopback-only"},
                )
        return await call_next(request)

    @app.get("/health")
    def health() -> dict[str, Any]:
        out = eng.health()
        out["server_revision"] = "fastapi-body-v4"
        from runtime.kv.backend import kv_native_build_sha

        sha = kv_native_build_sha()
        out["kv_native_build_sha"] = sha or ""
        return out

    @app.get("/ready")
    def ready():
        body = eng.health()
        if body.get("ready"):
            return body
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=503,
            content={
                "ready": False,
                "ready_reasons": body.get("ready_reasons", []),
                "ready_warnings": body.get("ready_warnings", []),
                "inference_state": body.get("inference_state"),
                "accepts_new_loads": body.get("accepts_new_loads"),
            },
        )

    @app.post("/v1/completions")
    def completions(req: CompletionRequest = Body()) -> dict[str, Any]:
        try:
            result = eng.generate(req.prompt, n_predict=req.n_predict)
            out: dict[str, Any] = {
                "content": result.content,
                "llama": result.llama,
            }
            if result.kv_decode_steps is not None:
                out["kv_decode_steps"] = result.kv_decode_steps
            return out
        except LlamaServerError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e

    def _llama_error_status(err: LlamaServerError) -> int:
        msg = str(err).lower()
        if any(
            x in msg
            for x in (
                "paused",
                "invalid",
                "unavailable",
                "misconfigured",
                "admission",
                "below admission",
                "inference-first",
                "gpu memory",
                "vram",
            )
        ):
            return 503
        return 502

    def _n_predict(options: dict) -> Optional[int]:
        """Output token limit; None means use llama-server default (unlimited)."""
        if "num_predict" not in options:
            return None
        try:
            n = int(options["num_predict"])
        except (TypeError, ValueError):
            return None
        return n if n > 0 else None

    def _request_num_ctx(
        opts: dict[str, Any],
        gguf: Any,
        *,
        explicit: Optional[int] = None,
    ) -> Optional[int]:
        # Why not resolve_vram_num_ctx alone: tools render and load must see the same
        # clamped ctx as _admit_one (Phase 13 resolve_num_ctx_for_request).
        out = eng.resolve_num_ctx_for_request(
            gguf, num_ctx=explicit, options=opts
        )
        if isinstance(out, tuple):
            return out[0]
        return out

    @app.post("/api/generate")
    async def api_generate(request: Request, req: OllamaGenerateRequest = Body()):
        started = log_request_in(
            "/api/generate",
            model=req.model,
            stream=bool(req.stream),
            queue=runtime_queue_snapshot(eng),
        )
        opts = dict(req.options)
        gguf = pop_gguf_path(opts)
        n_predict = _n_predict(opts)
        num_ctx = _request_num_ctx(opts, gguf)
        if req.stream:
            from runtime.server.disconnect_stream import ndjson_stream_on_disconnect

            queue_in = runtime_queue_snapshot(eng)
            meta: dict[str, Any] = {"status": 200, "done_reason": "", "error": ""}

            def _iter(cancel):
                try:
                    for chunk in eng.stream_generate(
                        req.prompt,
                        req.model,
                        n_predict=n_predict,
                        gguf=gguf,
                        num_ctx=num_ctx,
                        options=opts,
                        prefill_cancel=cancel,
                    ):
                        if chunk.get("done"):
                            meta["done_reason"] = str(
                                chunk.get("done_reason") or "stop"
                            )
                        yield chunk
                except LlamaServerError as e:
                    meta["status"] = _llama_error_status(e)
                    meta["error"] = "llama error"
                    yield {"error": str(e)}

            async def _gen():
                try:
                    async for line in ndjson_stream_on_disconnect(request, _iter):
                        yield line
                finally:
                    log_response_out(
                        "/api/generate",
                        started,
                        model=req.model,
                        stream=True,
                        status=meta["status"],
                        done_reason=meta["done_reason"]
                        or ("stop" if meta["status"] == 200 else ""),
                        error=meta["error"],
                        queue=runtime_queue_snapshot(eng),
                        queue_in=queue_in,
                    )

            return StreamingResponse(
                _gen(), media_type="application/x-ndjson"
            )
        try:
            from runtime.kv.native_decode_loop import PrefillAbortedError
            from runtime.server.disconnect_stream import run_sync_on_disconnect

            result = await run_sync_on_disconnect(
                request,
                lambda cancel: eng.generate(
                    req.prompt,
                    n_predict=n_predict,
                    gguf=gguf,
                    num_ctx=num_ctx,
                    options=opts,
                    prefill_cancel=cancel,
                ),
            )
            log_response_out(
                "/api/generate",
                started,
                model=req.model,
                stream=False,
                status=200,
                done_reason="stop",
                queue=runtime_queue_snapshot(eng),
                queue_in=runtime_queue_snapshot(eng),
            )
            return OllamaGenerateResponse(
                model=req.model,
                response=result.content,
                done=True,
                done_reason="stop",
                vram_num_ctx=result.vram_num_ctx,
                kv_decode_steps=result.kv_decode_steps,
            )
        except PrefillAbortedError:
            log_response_out(
                "/api/generate",
                started,
                model=req.model,
                stream=False,
                status=499,
                done_reason="cancelled",
                queue=runtime_queue_snapshot(eng),
                queue_in=runtime_queue_snapshot(eng),
            )
            raise HTTPException(status_code=499, detail="request cancelled") from None
        except LlamaServerError as e:
            status = (
                503
                if any(
                    x in str(e).lower()
                    for x in ("paused", "invalid", "unavailable", "misconfigured")
                )
                else 502
            )
            log_response_out(
                "/api/generate",
                started,
                model=req.model,
                stream=False,
                status=status,
                error=str(e),
            )
            raise HTTPException(status_code=status, detail=str(e)) from e
        except Exception as e:
            log_response_out(
                "/api/generate",
                started,
                model=req.model,
                stream=False,
                status=500,
                error=str(e),
            )
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/api/chat")
    async def api_chat(request: Request, req: OllamaChatRequest = Body()):
        from runtime.server.chat_tools import (
            ToolParseUnavailableError,
            normalize_tools,
            parse_completion_tool_calls,
            resolve_tools_chat_prompt,
            stream_tool_chat_chunks,
        )
        from runtime.server.runtime_chat import chat_needs_legacy, messages_to_prompt

        started = log_request_in(
            "/api/chat",
            model=req.model,
            stream=bool(req.stream),
            queue=runtime_queue_snapshot(eng),
        )

        if chat_needs_legacy(
            list(req.messages),
            tools=req.tools,
            logprobs=bool(req.logprobs),
            think=req.think,
        ):
            raise HTTPException(
                status_code=501,
                detail="request requires legacy runner (vision/logprobs/think)",
            )
        opts = dict(req.options)
        gguf = pop_gguf_path(opts)
        n_predict = _n_predict(opts)
        num_ctx = _request_num_ctx(opts, gguf)
        tools = normalize_tools(list(req.tools))
        tool_tag = "{"
        tools_meta: dict[str, Any] = {}
        if tools:
            prompt, tool_tag, tools_meta = resolve_tools_chat_prompt(
                req.model,
                list(req.messages),
                tools,
                think=req.think,
                num_ctx=num_ctx,
                n_predict=n_predict,
            )
        else:
            prompt = messages_to_prompt(list(req.messages))
        if not prompt:
            raise HTTPException(status_code=400, detail="empty messages")
        if req.stream:
            from runtime.server.disconnect_stream import ndjson_stream_on_disconnect

            queue_in = runtime_queue_snapshot(eng)
            meta: dict[str, Any] = {"status": 200, "done_reason": "", "error": ""}

            def _iter(cancel):
                try:
                    if tools:
                        it = stream_tool_chat_chunks(
                            eng,
                            prompt,
                            req.model,
                            tools,
                            n_predict=n_predict,
                            gguf=gguf,
                            num_ctx=num_ctx,
                            options=opts,
                            tag=tool_tag,
                            messages=list(req.messages),
                            think=req.think,
                            tools_meta=tools_meta,
                            prefill_cancel=cancel,
                        )
                    else:
                        it = eng.stream_chat(
                            prompt,
                            req.model,
                            n_predict=n_predict,
                            gguf=gguf,
                            num_ctx=num_ctx,
                            options=opts,
                            prefill_cancel=cancel,
                        )
                    for chunk in it:
                        if isinstance(chunk, dict) and chunk.get("done"):
                            meta["done_reason"] = str(
                                chunk.get("done_reason") or "stop"
                            )
                        yield chunk
                except ToolParseUnavailableError as e:
                    meta["status"] = 503
                    meta["error"] = str(e)
                    yield {"error": str(e)}
                except LlamaServerError as e:
                    meta["status"] = _llama_error_status(e)
                    meta["error"] = "inference error"
                    yield {"error": str(e)}

            async def _gen():
                try:
                    async for line in ndjson_stream_on_disconnect(request, _iter):
                        yield line
                finally:
                    log_response_out(
                        "/api/chat",
                        started,
                        model=req.model,
                        stream=True,
                        status=meta["status"],
                        done_reason=meta["done_reason"]
                        or ("stop" if meta["status"] == 200 else ""),
                        error=meta["error"],
                        queue=runtime_queue_snapshot(eng),
                        queue_in=queue_in,
                    )

            return StreamingResponse(
                _gen(), media_type="application/x-ndjson"
            )
        try:
            from runtime.kv.native_decode_loop import PrefillAbortedError
            from runtime.server.disconnect_stream import run_sync_on_disconnect

            result = await run_sync_on_disconnect(
                request,
                lambda cancel: eng.generate(
                    prompt,
                    n_predict=n_predict,
                    gguf=gguf,
                    num_ctx=num_ctx,
                    options=opts,
                    prefill_cancel=cancel,
                ),
            )
            msg: dict[str, Any] = {"role": "assistant", "content": result.content}
            done_reason = "stop"
            if tools:
                calls, content = parse_completion_tool_calls(
                    result.content,
                    tools,
                    tag=tool_tag,
                    model=req.model,
                    messages=list(req.messages),
                    think=req.think,
                    tools_meta=tools_meta,
                )
                if calls:
                    msg["tool_calls"] = calls
                    msg["content"] = content
                    done_reason = "tool_calls"
            out: dict[str, Any] = {
                "model": req.model,
                "message": msg,
                "done": True,
                "done_reason": done_reason,
            }
            if result.vram_num_ctx:
                out["vram_num_ctx"] = result.vram_num_ctx
            if result.kv_decode_steps is not None:
                out["kv_decode_steps"] = result.kv_decode_steps
            log_response_out(
                "/api/chat",
                started,
                model=req.model,
                stream=False,
                status=200,
                done_reason=done_reason,
                queue=runtime_queue_snapshot(eng),
                queue_in=runtime_queue_snapshot(eng),
            )
            return out
        except PrefillAbortedError:
            log_response_out(
                "/api/chat",
                started,
                model=req.model,
                stream=False,
                status=499,
                done_reason="cancelled",
                queue=runtime_queue_snapshot(eng),
                queue_in=runtime_queue_snapshot(eng),
            )
            raise HTTPException(status_code=499, detail="request cancelled") from None
        except ToolParseUnavailableError as e:
            log_response_out(
                "/api/chat",
                started,
                model=req.model,
                stream=False,
                status=503,
                error=str(e),
            )
            raise HTTPException(status_code=503, detail=str(e)) from e
        except LlamaServerError as e:
            status = _llama_error_status(e)
            log_response_out(
                "/api/chat",
                started,
                model=req.model,
                stream=False,
                status=status,
                error=str(e),
            )
            raise HTTPException(status_code=status, detail=str(e)) from e

    @app.post("/v1/chat/completions")
    async def v1_chat_completions(request: Request, payload: dict[str, Any] = Body()):
        from runtime.server.chat_tools import ToolParseUnavailableError
        from runtime.server.disconnect_stream import sse_stream_on_disconnect
        from runtime.server.openai_v1 import (
            run_v1_chat_completion,
            stream_openai_sse,
            v1_needs_legacy,
        )

        if v1_needs_legacy(payload):
            raise HTTPException(
                status_code=501,
                detail="request requires legacy runner (vision/logprobs/think)",
            )
        if payload.get("stream"):

            async def _sse():
                async for line in sse_stream_on_disconnect(
                    request,
                    lambda cancel: stream_openai_sse(
                        eng, payload, prefill_cancel=cancel
                    ),
                ):
                    yield line

            return StreamingResponse(
                _sse(), media_type="text/event-stream"
            )
        try:
            from runtime.kv.native_decode_loop import PrefillAbortedError
            from runtime.server.disconnect_stream import run_sync_on_disconnect

            return await run_sync_on_disconnect(
                request,
                lambda cancel: run_v1_chat_completion(
                    eng, payload, prefill_cancel=cancel
                ),
            )
        except PrefillAbortedError:
            raise HTTPException(status_code=499, detail="request cancelled") from None
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except ToolParseUnavailableError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except LlamaServerError as e:
            raise HTTPException(status_code=_llama_error_status(e), detail=str(e)) from e

    @app.post("/internal/training-handoff")
    def training_handoff() -> dict[str, str]:
        state = eng.training_handoff()
        return {"status": "ok", "inference_state": state.value}

    class TrainingGpuBusyBody(BaseModel):
        busy: bool = True

    @app.post("/internal/training-gpu-busy")
    def training_gpu_busy(body: TrainingGpuBusyBody = Body()) -> dict[str, object]:
        """Go training policy: training holds GPU (VRAM reserve before full handoff)."""
        eng.coordinator.set_go_training_gpu_busy(body.busy)
        eng.invalidate_health_cache()
        return {
            "status": "ok",
            "go_training_gpu_busy": body.busy,
        }

    @app.post("/internal/inference/resume")
    def inference_resume() -> dict[str, str]:
        state = eng.resume_inference()
        return {"status": "ok", "inference_state": state.value}

    @app.post("/internal/go-coordination")
    def go_coordination(body: dict[str, Any] = Body()) -> dict[str, str]:
        """Go daemon pushes training defer / policy snapshot for /health."""
        from runtime.go_coordination import update_go_coordination

        update_go_coordination(body)
        resumed = eng.maybe_auto_resume_inference()
        eng.invalidate_health_cache()
        out: dict[str, str] = {"status": "ok"}
        if resumed:
            out["inference_state"] = eng.coordinator.state.value
            out["auto_resumed"] = "true"
        return out

    class VramEstimateBody(BaseModel):
        gguf: str
        num_ctx: Optional[int] = None
        options: dict[str, Any] = Field(default_factory=dict)

    class TokenizeBody(BaseModel):
        gguf: str
        text: str
        add_special: bool = True

    class InternalBatchGenerateBody(BaseModel):
        """Loopback batch generate for Phase 15 GPU sign-off (v27–v30)."""

        prompts: list[str] = Field(min_length=1, max_length=8)
        n_predict: int = Field(default=8, ge=1, le=256)
        max_admit: int = Field(default=4, ge=1, le=8)
        stream: bool = False
        options: Optional[dict[str, Any]] = None

    @app.post("/internal/tokenize")
    def internal_tokenize(body: TokenizeBody = Body()) -> dict[str, Any]:
        """Loopback-only tokenizer for Go /internal/render-chat (Phase 14).

        Why vocab-only loads: render-chat needs token ids without pulling full weights
        onto GPU for every truncation attempt. engine.tokenize_gguf_text reuses the
        loaded worker when GGUF matches; otherwise caches LlamaVocabSession (max 4).
        """
        from pathlib import Path

        raw_gguf = body.gguf.strip()
        if not raw_gguf:
            raise HTTPException(status_code=400, detail="gguf path required")
        if not body.text:
            return {"tokens": [], "count": 0}
        path = Path(raw_gguf)
        if not path.is_absolute():
            path = path.resolve()
        try:
            tokens = eng.tokenize_gguf_text(
                path, body.text, add_special=body.add_special
            )
        except LlamaServerError as e:
            raise HTTPException(
                status_code=_llama_error_status(e), detail=str(e)
            ) from e
        return {"tokens": tokens, "count": len(tokens)}

    @app.post("/internal/vram-estimate")
    def internal_vram_estimate(body: VramEstimateBody = Body()) -> dict[str, Any]:
        """Loopback-only VRAM estimate for a GGUF path (no load)."""
        from pathlib import Path

        raw = body.gguf.strip()
        if not raw:
            raise HTTPException(status_code=400, detail="gguf path required")
        path = Path(raw)
        if not path.is_absolute():
            path = path.resolve()
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"gguf not found: {path}")
        opts = dict(body.options)
        if body.num_ctx is not None:
            opts["num_ctx"] = body.num_ctx
        est, budget = eng.vram_estimate_and_budget(
            path,
            num_ctx=body.num_ctx,
            options=opts or None,
        )
        if est is None:
            raise HTTPException(status_code=500, detail="vram estimate failed")
        return {"vram_estimate": est, "vram_budget": budget}

    @app.post("/internal/generate-batch")
    def internal_generate_batch(body: InternalBatchGenerateBody = Body()):
        """Loopback-only batched generate (Phase 15 v27–v30 GPU sign-off).

        WHY internal, not public /api/generate: batch admission policy and streaming
        NDJSON shape are still evolving; this endpoint exercises the real engine
        path (generate_batch / stream_generate_batch) without committing to an
        external OpenAI-compatible contract. GPU smokes call it from localhost only.
        """
        import json

        opts = dict(body.options or {})
        try:
            if body.stream:

                def _gen():
                    for chunk in eng.stream_generate_batch(
                        body.prompts,
                        n_predict=body.n_predict,
                        max_admit=body.max_admit,
                        options=opts,
                    ):
                        yield json.dumps(chunk) + "\n"

                return StreamingResponse(
                    _gen(), media_type="application/x-ndjson"
                )
            results = eng.generate_batch(
                body.prompts,
                n_predict=body.n_predict,
                max_admit=body.max_admit,
                options=opts,
            )
            return {
                "results": [
                    {
                        "request_id": r.request_id,
                        "content": r.content,
                        "kv_decode_steps": r.kv_decode_steps,
                    }
                    for r in results
                ]
            }
        except LlamaServerError as e:
            raise HTTPException(
                status_code=_llama_error_status(e), detail=str(e)
            ) from e

    @app.get("/internal/kv-snapshot")
    def internal_kv_snapshot() -> dict[str, Any]:
        """Loopback-only KV scheduler/bind snapshot (Phase 15 debug)."""
        return eng.kv_snapshot()

    return app
