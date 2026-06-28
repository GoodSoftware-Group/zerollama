"""Copy KV prefix between llama sequences (cross-slot Radix seed).

WHY this module exists: L3 pins each ``prompt_cache_key`` to a fixed llama-server
slot. Two agents with the same system prompt but different keys land on different
slots and would repeat full prefill unless we copy donor KV into the cold target
before decode. In-process uses ``llama_memory_seq_cp``; subprocess uses patch 0017
``POST /kv/seq-copy`` because Python cannot ctypes into the llama-server child.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from runtime.kv.radix_prefix_share import RadixSharePlan

_log = logging.getLogger(__name__)


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
        ctypes.c_int32(0),
        ctypes.c_int32(-1),
    )
    lib.llama_memory_seq_cp(
        mem,
        ctypes.c_int32(int(source_slot)),
        ctypes.c_int32(int(target_slot)),
        ctypes.c_int32(0),
        ctypes.c_int32(int(pos_end)),
    )
    return True


def copy_sequence_prefix_subprocess(
    base_url: str,
    *,
    source_slot: int,
    target_slot: int,
    pos_end: int,
    timeout: float = 30.0,
) -> bool:
    """POST ``/kv/seq-copy`` on llama-server (requires vendor patch 0017).

    WHY HTTP not ctypes: default backend runs llama-server in a child process;
    ggml memory lives there. 404/501 → rebuild vendor binary (bare sibling lacks route).
    """
    if pos_end <= 0 or source_slot < 0 or target_slot < 0:
        return False
    if source_slot == target_slot:
        return True
    url = base_url.rstrip("/") + "/kv/seq-copy"
    body = json.dumps(
        {
            "src_slot": int(source_slot),
            "dst_slot": int(target_slot),
            "pos_end": int(pos_end),
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
        )
    return False
