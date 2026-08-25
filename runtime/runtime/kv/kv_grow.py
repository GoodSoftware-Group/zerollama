"""In-place llama.cpp n_ctx grow/shrink (no llama_init_from_model restart).

WHY: ``POST /kv/grow`` / ``llama_n_ctx_grow`` copy used KV into a larger buffer.
``POST /kv/shrink`` / ``llama_n_ctx_shrink`` pack live cells then cut the buffer.
Fail closed → caller keeps the current size (or restarts).
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

_log = logging.getLogger(__name__)


def _post_kv_n_ctx(base_url: str, route: str, n_ctx: int, *, timeout: float = 120.0) -> dict[str, Any]:
    url = base_url.rstrip("/") + route
    payload = json.dumps({"n_ctx": int(n_ctx)}).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, method="POST", headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def post_kv_grow(base_url: str, n_ctx: int, *, timeout: float = 120.0) -> dict[str, Any]:
    return _post_kv_n_ctx(base_url, "/kv/grow", n_ctx, timeout=timeout)


def post_kv_shrink(base_url: str, n_ctx: int, *, timeout: float = 120.0) -> dict[str, Any]:
    return _post_kv_n_ctx(base_url, "/kv/shrink", n_ctx, timeout=timeout)


def _try_worker_hook(worker: Any, method: str, n_ctx: int, route: str) -> bool:
    if worker is None or n_ctx <= 0:
        return False
    fn = getattr(worker, method, None)
    if callable(fn):
        try:
            return bool(fn(int(n_ctx)))
        except Exception:
            _log.warning("kv %s worker hook failed", method, exc_info=True)
            return False
    base = getattr(worker, "base_url", None)
    kind = getattr(worker, "__class__", type(worker)).__name__
    if kind == "LlamaServerProcess" and base:
        post = post_kv_grow if route == "/kv/grow" else post_kv_shrink
        try:
            out = post(str(base), int(n_ctx))
            return bool(out.get("ok", True))
        except urllib.error.HTTPError as e:
            _log.info("POST %s HTTP %s", route, e.code)
            return False
        except Exception:
            _log.warning("POST %s failed", route, exc_info=True)
            return False
    return False


def try_grow_worker(worker: Any, n_ctx: int) -> bool:
    """Grow the live worker's KV. Returns False when unsupported or failed."""
    return _try_worker_hook(worker, "grow_n_ctx", n_ctx, "/kv/grow")


def try_shrink_worker(worker: Any, n_ctx: int) -> bool:
    """Shrink the live worker's KV. Fails if live tokens do not fit."""
    return _try_worker_hook(worker, "shrink_n_ctx", n_ctx, "/kv/shrink")
