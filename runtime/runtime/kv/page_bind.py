"""Phase 15 v8 — PA block pool → llama KV tensor page bind (not implemented).

v0–v7 export logical page tables (`kv_forward_plans`) and bind llama sequence/slot ids.
v8 will map ``block_ids[i]`` to physical KV tensor pages when llama exposes a stable API.
"""

from __future__ import annotations

from typing import Any


def page_bind_health(*, native_ext_available: bool) -> dict[str, Any]:
    """Operator-facing status for tensor/page bind readiness."""
    return {
        "available": False,
        "status": "not_implemented",
        "reason": (
            "llama public API lacks stable paged KV tensor handles; "
            "use kv_forward_plans for logical page tables"
        ),
        "native_ext_available": native_ext_available,
    }
