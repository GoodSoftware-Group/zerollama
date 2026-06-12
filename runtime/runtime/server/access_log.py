"""Inference request/response access logging (stderr, visible under zerollama serve)."""

from __future__ import annotations

import time
from typing import Any

from runtime.logutil import get_logger

_log = get_logger("access")


def runtime_queue_snapshot(eng: Any) -> dict[str, int | bool]:
    return {
        "rt_waiting": len(eng.scheduler.waiting),
        "rt_running": len(eng.scheduler.running),
        "rt_llama_loaded": bool(
            eng._server is not None and eng._server.is_running()
        ),
    }


def _format_queue(q: dict[str, int | bool]) -> str:
    return " ".join(f"{k}={v}" for k, v in q.items())


def log_request_in(
    route: str,
    *,
    model: str = "",
    stream: bool = False,
    queue: dict[str, int | bool] | None = None,
) -> float:
    started = time.monotonic()
    parts = [f"inference request in route={route}", f"stream={stream}"]
    if model:
        parts.append(f"model={model}")
    if queue:
        parts.append(_format_queue(queue))
    _log.info(" ".join(parts))
    return started


def log_response_out(
    route: str,
    started: float,
    *,
    model: str = "",
    stream: bool = False,
    status: int = 200,
    done_reason: str = "",
    eval_count: int | None = None,
    prompt_eval_count: int | None = None,
    error: str = "",
    queue: dict[str, int | bool] | None = None,
    queue_in: dict[str, int | bool] | None = None,
) -> None:
    duration_ms = int((time.monotonic() - started) * 1000)
    parts = [
        f"inference response out route={route}",
        f"stream={stream}",
        f"status={status}",
        f"duration={duration_ms}ms",
    ]
    if model:
        parts.append(f"model={model}")
    if queue:
        parts.append(_format_queue(queue))
    if queue_in and queue and queue_in != queue:
        parts.append(
            f"rt_waiting_in={queue_in.get('rt_waiting', 0)} "
            f"rt_running_in={queue_in.get('rt_running', 0)}"
        )
    if done_reason:
        parts.append(f"done_reason={done_reason}")
    if prompt_eval_count is not None and prompt_eval_count > 0:
        parts.append(f"prompt_eval_count={prompt_eval_count}")
    if eval_count is not None and eval_count > 0:
        parts.append(f"eval_count={eval_count}")
    if error:
        parts.append(f"error={error}")
    _log.info(" ".join(parts))
