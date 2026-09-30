#!/usr/bin/env python3
"""gliner-decide-server — Fastino GLiNER2.5-Decide sibling (GD1).

Dual wire:
  POST /v1/systemone  — Jev DecisionsRequest → DecisionsResponse (calibrated:false)
  POST /v1/gliner-decide — mechanical {text, schema} → classify_text result

Lab only. Docs: docs/gliner-decide.md
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="gliner-decide-server", version="0.1.0")

_MODEL = None
_MODEL_LOCK = threading.Lock()
_MODEL_ID = os.environ.get("GLINER_DECIDE_MODEL", "fastino/GLiNER2.5-Decide")
_DEVICE = os.environ.get("GLINER_DECIDE_DEVICE", "cpu")


def _get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        from gliner2 import AutoExtractor

        kwargs: dict[str, Any] = {}
        # gliner2 device knobs vary by version; try common patterns.
        if _DEVICE and _DEVICE != "cpu":
            kwargs["device"] = _DEVICE
        try:
            _MODEL = AutoExtractor.from_pretrained(_MODEL_ID, **kwargs)
        except TypeError:
            _MODEL = AutoExtractor.from_pretrained(_MODEL_ID)
            if hasattr(_MODEL, "to") and _DEVICE:
                try:
                    _MODEL.to(_DEVICE)
                except Exception:
                    pass
        return _MODEL


def _state_to_text(state: Any) -> str:
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _choice_labels(criteria: Any) -> list[str]:
    if isinstance(criteria, dict):
        return [str(k) for k in criteria.keys()]
    if isinstance(criteria, list):
        return [str(x) for x in criteria]
    raise ValueError("choice criteria must be object or list")


def _score_labels(criteria: Any) -> list[str]:
    if isinstance(criteria, list):
        # Prefer stable level ids matching Laya score legend keys.
        return [str(i) for i in range(len(criteria))]
    if isinstance(criteria, dict):
        return [str(k) for k in criteria.keys()]
    raise ValueError("score criteria must be list or object")


def _build_schema(questions: dict[str, Any]) -> dict[str, Any]:
    schema: dict[str, Any] = {}
    for qid, q in questions.items():
        if not isinstance(q, dict):
            raise ValueError(f"question {qid!r} must be an object")
        t = str(q.get("type", "")).lower().strip()
        crit = q.get("criteria")
        if t == "choice":
            schema[qid] = _choice_labels(crit)
        elif t == "score":
            schema[qid] = _score_labels(crit)
        elif t == "noul":
            # Fastino handoff-style yes/no; map P(yes) → noul.
            schema[qid] = ["yes", "no"]
        else:
            raise ValueError(f"unsupported question type {t!r} for {qid!r}")
    return schema


def _normalize_task_result(raw: Any) -> tuple[str | None, float | None, dict[str, float] | None]:
    """Return (label, confidence, probabilities) from classify_text task output."""
    if isinstance(raw, str):
        return raw, None, None
    if not isinstance(raw, dict):
        return None, None, None
    label = raw.get("label") or raw.get("category") or raw.get("value")
    if isinstance(label, list):
        label = label[0] if label else None
    conf = raw.get("confidence")
    probs = raw.get("probabilities")
    if isinstance(probs, dict):
        probs = {str(k): float(v) for k, v in probs.items()}
    else:
        probs = None
    if conf is not None:
        conf = float(conf)
    if label is not None:
        label = str(label)
    return label, conf, probs


def _classify(text: str, schema: dict[str, Any], include_confidence: bool = True) -> Any:
    model = _get_model()
    try:
        return model.classify_text(text, schema, include_confidence=include_confidence)
    except TypeError:
        return model.classify_text(text, schema)


def _answers_from_classify(
    questions: dict[str, Any], result: Any
) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("classify_text returned non-object")
    answers: dict[str, Any] = {}
    for qid, q in questions.items():
        t = str(q.get("type", "")).lower().strip()
        raw = result.get(qid)
        label, conf, probs = _normalize_task_result(raw)
        if t == "choice":
            keys = _choice_labels(q.get("criteria"))
            if probs is None and label is not None:
                probs = {k: (1.0 if k == label else 0.0) for k in keys}
            if label is None and probs:
                label = max(probs.items(), key=lambda kv: kv[1])[0]
            if conf is None and probs and label in probs:
                conf = float(probs[label])
            answers[qid] = {
                "type": "choice",
                "choice": label or "",
                "probabilities": probs or {},
                "confidence": conf if conf is not None else 0.0,
                "calibrated": False,
            }
        elif t == "score":
            n = len(_score_labels(q.get("criteria")))
            if probs is None and label is not None:
                probs = {str(i): (1.0 if str(i) == str(label) else 0.0) for i in range(n)}
            if probs is None:
                probs = {str(i): 0.0 for i in range(n)}
            # Expected level index (Laya-shaped).
            exp = sum(float(i) * float(probs.get(str(i), 0.0)) for i in range(n))
            if conf is None and probs:
                conf = max(probs.values()) if probs else 0.0
            legend = {}
            crit = q.get("criteria")
            if isinstance(crit, list):
                for i, c in enumerate(crit):
                    legend[str(i)] = str(c)
            answers[qid] = {
                "type": "score",
                "score": round(exp, 4),
                "legend": legend,
                "probabilities": probs,
                "confidence": float(conf) if conf is not None else 0.0,
                "calibrated": False,
            }
        elif t == "noul":
            noul = 0.0
            if probs and "yes" in probs:
                noul = float(probs["yes"])
            elif label is not None:
                noul = 1.0 if str(label).lower() in ("yes", "true", "1") else 0.0
            answers[qid] = {
                "type": "noul",
                "noul": round(noul, 4),
                "confidence": round(max(noul, 1.0 - noul), 4),
                "calibrated": False,
            }
    return answers


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": _MODEL_ID,
        "device": _DEVICE,
        "loaded": _MODEL is not None,
    }


@app.post("/v1/gliner-decide")
async def gliner_decide(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(400, f"invalid json: {e}") from e
    text = body.get("text")
    schema = body.get("schema")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(400, "text required")
    if not isinstance(schema, dict) or not schema:
        raise HTTPException(400, "schema object required")
    include_confidence = bool(body.get("include_confidence", True))
    try:
        result = _classify(text, schema, include_confidence=include_confidence)
    except Exception as e:
        raise HTTPException(500, f"classify_text failed: {e}") from e
    return {
        "model": body.get("model") or "gliner-decide",
        "result": result,
        "engine": {"model_id": _MODEL_ID, "device": _DEVICE},
    }


@app.post("/v1/systemone")
@app.post("/v1/decisions")
async def systemone(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(400, f"invalid json: {e}") from e
    model_name = body.get("model") or "gliner-decide"
    state = body.get("state")
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise HTTPException(400, "questions object required")
    text = _state_to_text(state)
    if not text.strip():
        raise HTTPException(400, "state text required")
    try:
        schema = _build_schema(questions)
        result = _classify(text, schema, include_confidence=True)
        answers = _answers_from_classify(questions, result)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, f"decide failed: {e}") from e
    return JSONResponse(
        {
            "model": model_name,
            "answers": answers,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
    )


def main():
    import uvicorn

    host = os.environ.get("GLINER_DECIDE_HOST", "127.0.0.1")
    port = int(os.environ.get("GLINER_DECIDE_PORT", "18098"))
    if port in (11434, 8081, 8080):
        raise SystemExit(f"refusing reserved port {port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
