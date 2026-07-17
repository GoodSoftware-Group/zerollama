"""Copy KV prefix between llama sequences (cross-slot Radix seed).

WHY this module exists: L3 pins each ``prompt_cache_key`` to a fixed llama-server
slot. Two agents with the same system prompt but different keys land on different
slots and would repeat full prefill unless we copy donor KV into the cold target
before decode. In-process uses ``llama_memory_seq_cp``; subprocess uses patch 0017
``POST /kv/seq-copy`` because Python cannot ctypes into the llama-server child.

Phase 15 v52: with ``ZEROLLAMA_KV_UNIFIED=1``, in-process multi-seq uses one
stream so ``seq_cp`` is metadata-only (cell mask share) — no buffer copy.
Cross-stream (default) still requires a full-buffer ``seq_cp`` then trim.

Media (Jul 2026): subprocess ``allow_media`` (default on via
``ZEROLLAMA_RADIX_MEDIA_SEQ_COPY``) lets llama-server clone mtmd chunks with
``keep_first``; set ``0`` for text-only clamp before first media.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

from runtime.kv.radix_prefix_share import RadixSharePlan

_log = logging.getLogger(__name__)


def seq_cp_mode() -> str:
    """``metadata`` when unified KV shares one stream; else ``buffer_copy``."""
    try:
        from runtime.env import kv_unified_enabled

        return "metadata" if kv_unified_enabled() else "buffer_copy"
    except Exception:
        return "buffer_copy"


def radix_media_seq_copy_enabled() -> bool:
    """When true, POST /kv/seq-copy sets allow_media (clone mtmd chunks)."""
    raw = os.environ.get("ZEROLLAMA_RADIX_MEDIA_SEQ_COPY", "1").strip().lower()
    return raw not in ("0", "false", "off", "no", "text", "text_only", "text-only")


def copy_sequence_prefix_inprocess(
    lib: Any,
    ctx: Any,
    *,
    source_slot: int,
    target_slot: int,
    pos_end: int,
) -> bool:
    """Copy ``[0, pos_end)`` KV from ``source_slot`` → ``target_slot``.

    WHY clear target first: ``seq_cp`` appends into existing KV; a cold slot may
    hold stale bytes from a prior session on the same ``id_slot``.

    WHY ``p0/p1 = -1,-1`` then trim (v52 harden): cross-stream ``seq_cp`` asserts
    full-buffer ranges; partial ``p1=pos_end`` aborts when ``pos_end < kv_size``.
    Matches llama-server patch 0017 / ``copy_state_to`` (see ggml-org#13833).
    Under ``kv_unified``, both slots share one stream so ``seq_cp`` is metadata-
    only; full-range + trim remains correct.

    Note: in-process path has no mtmd prompt map — media-aware metadata clone is
    subprocess-only (llama-server ``server_tokens``).
    """
    if pos_end <= 0 or source_slot < 0 or target_slot < 0:
        return False
    if source_slot == target_slot:
        return True
    import ctypes

    mem = lib.llama_get_memory(ctx)
    if not mem:
        return False
    lib.llama_memory_seq_rm(
        mem,
        ctypes.c_int32(int(target_slot)),
        ctypes.c_int32(-1),
        ctypes.c_int32(-1),
    )
    lib.llama_memory_seq_cp(
        mem,
        ctypes.c_int32(int(source_slot)),
        ctypes.c_int32(int(target_slot)),
        ctypes.c_int32(-1),
        ctypes.c_int32(-1),
    )
    # Trim to requested prefix when full-seq copy is longer than pos_end.
    lib.llama_memory_seq_rm(
        mem,
        ctypes.c_int32(int(target_slot)),
        ctypes.c_int32(int(pos_end)),
        ctypes.c_int32(-1),
    )
    return True


def copy_sequence_prefix_subprocess(
    base_url: str,
    *,
    source_slot: int,
    target_slot: int,
    pos_end: int,
    timeout: float = 30.0,
    allow_media: bool | None = None,
) -> bool:
    """POST ``/kv/seq-copy`` on llama-server (requires vendor patch 0017 + 0090 media).

    WHY HTTP not ctypes: default backend runs llama-server in a child process;
    ggml memory lives there. 404/501 → rebuild vendor binary (bare sibling lacks route).

    ``allow_media`` defaults from ``ZEROLLAMA_RADIX_MEDIA_SEQ_COPY`` (on). Server
    clones mtmd chunks via keep_first; false clamps to pure-text before first media.
    """
    if pos_end <= 0 or source_slot < 0 or target_slot < 0:
        return False
    if source_slot == target_slot:
        return True
    if allow_media is None:
        allow_media = radix_media_seq_copy_enabled()
    url = base_url.rstrip("/") + "/kv/seq-copy"
    body = json.dumps(
        {
            "src_slot": int(source_slot),
            "dst_slot": int(target_slot),
            "pos_end": int(pos_end),
            "allow_media": bool(allow_media),
        }
    ).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        if e.code in (404, 501):
            _log.debug("radix seq-copy unsupported on llama-server: HTTP %s", e.code)
            return False
        return False
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return False
    return bool(data.get("ok"))


def execute_radix_share_plan(
    plan: RadixSharePlan,
    *,
    inprocess_lib: Any | None = None,
    inprocess_ctx: Any | None = None,
    subprocess_base_url: str | None = None,
    allow_media: bool | None = None,
) -> bool:
    """Execute a RadixSharePlan on whichever backend is active.

    WHY in-process vs subprocess: default backend runs llama-server in a child;
    only one path has live ggml memory. Both clear target slot before copy.
    """
    if inprocess_lib is not None and inprocess_ctx is not None:
        return copy_sequence_prefix_inprocess(
            inprocess_lib,
            inprocess_ctx,
            source_slot=plan.source_slot,
            target_slot=plan.target_slot,
            pos_end=plan.copy_tokens,
        )
    if subprocess_base_url:
        return copy_sequence_prefix_subprocess(
            subprocess_base_url,
            source_slot=plan.source_slot,
            target_slot=plan.target_slot,
            pos_end=plan.copy_tokens,
            allow_media=allow_media,
        )
    return False
