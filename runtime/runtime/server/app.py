"""HTTP sidecar (Phase 1–2).

Why FastAPI handlers use ``req: Model = Body()`` (no ``from __future__ import annotations``):
postponed annotations make FastAPI treat parameters as query fields (422 on ``query.req``).
Per-request weights: ``options.gguf`` (from Go manifest lookup) or ``LLAMA_MODEL`` env fallback.
"""

from typing import Any

from runtime.config import RuntimeConfig
from runtime.engine import InferenceEngine
from runtime.worker.llama_server import LlamaServerError


def create_app(
    config: RuntimeConfig | None = None,
    engine: InferenceEngine | None = None,
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

    eng = engine or InferenceEngine(config)

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
        out["server_revision"] = "fastapi-body-v3"
        return out

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

    def _n_predict(options: dict) -> int | None:
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
        explicit: int | None = None,
    ) -> int | None:
        # Why not resolve_vram_num_ctx alone: tools render and load must see the same
        # clamped ctx as _admit_one (Phase 13 resolve_num_ctx_for_request).
        out = eng.resolve_num_ctx_for_request(
            gguf, num_ctx=explicit, options=opts
        )
        if isinstance(out, tuple):
            return out[0]
        return out

    @app.post("/api/generate")
    def api_generate(req: OllamaGenerateRequest = Body()):
        opts = dict(req.options)
        gguf = pop_gguf_path(opts)
        n_predict = _n_predict(opts)
        num_ctx = _request_num_ctx(opts, gguf)
        if req.stream:
            import json

            def _gen():
                try:
                    for chunk in eng.stream_generate(
                        req.prompt,
                        req.model,
                        n_predict=n_predict,
                        gguf=gguf,
                        num_ctx=num_ctx,
                        options=opts,
                    ):
                        yield json.dumps(chunk) + "\n"
                except LlamaServerError as e:
                    yield json.dumps({"error": str(e)}) + "\n"

            return StreamingResponse(
                _gen(), media_type="application/x-ndjson"
            )
        try:
            result = eng.generate(
                req.prompt,
                n_predict=n_predict,
                gguf=gguf,
                num_ctx=num_ctx,
                options=opts,
            )
            return OllamaGenerateResponse(
                model=req.model,
                response=result.content,
                done=True,
                done_reason="stop",
                vram_num_ctx=result.vram_num_ctx,
                kv_decode_steps=result.kv_decode_steps,
            )
        except LlamaServerError as e:
            status = (
                503
                if any(
                    x in str(e).lower()
                    for x in ("paused", "invalid", "unavailable", "misconfigured")
                )
                else 502
            )
            raise HTTPException(status_code=status, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/api/chat")
    def api_chat(req: OllamaChatRequest = Body()):
        from runtime.server.chat_tools import (
            ToolParseUnavailableError,
            normalize_tools,
            parse_completion_tool_calls,
            resolve_tools_chat_prompt,
            stream_tool_chat_chunks,
        )
        from runtime.server.runtime_chat import chat_needs_legacy, messages_to_prompt

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
            import json

            def _gen():
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
                        )
                    else:
                        it = eng.stream_chat(
                            prompt,
                            req.model,
                            n_predict=n_predict,
                            gguf=gguf,
                            num_ctx=num_ctx,
                            options=opts,
                        )
                    for chunk in it:
                        yield json.dumps(chunk) + "\n"
                except ToolParseUnavailableError as e:
                    yield json.dumps({"error": str(e)}) + "\n"
                except LlamaServerError as e:
                    yield json.dumps({"error": str(e)}) + "\n"

            return StreamingResponse(
                _gen(), media_type="application/x-ndjson"
            )
        try:
            result = eng.generate(
                prompt,
                n_predict=n_predict,
                gguf=gguf,
                num_ctx=num_ctx,
                options=opts,
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
            return out
        except ToolParseUnavailableError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except LlamaServerError as e:
            raise HTTPException(status_code=_llama_error_status(e), detail=str(e)) from e

    @app.post("/v1/chat/completions")
    def v1_chat_completions(payload: dict[str, Any] = Body()):
        from runtime.server.chat_tools import ToolParseUnavailableError
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
            return StreamingResponse(
                stream_openai_sse(eng, payload),
                media_type="text/event-stream",
            )
        try:
            return run_v1_chat_completion(eng, payload)
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
        return {"status": "ok"}

    class VramEstimateBody(BaseModel):
        gguf: str
        num_ctx: int | None = None
        options: dict[str, Any] = Field(default_factory=dict)

    class TokenizeBody(BaseModel):
        gguf: str
        text: str
        add_special: bool = True

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

    @app.get("/internal/kv-snapshot")
    def internal_kv_snapshot() -> dict[str, Any]:
        """Loopback-only KV scheduler/bind snapshot (Phase 15 debug)."""
        return eng.kv_snapshot()

    return app
