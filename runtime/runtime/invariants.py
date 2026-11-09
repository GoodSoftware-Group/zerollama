"""Wire invariants shared by /v1 chat, /api/chat, and llama-server reload.

Call sites use these functions. A later change that re-derives the condition
inline is how the rule gets dropped.
"""

from __future__ import annotations

from typing import Any

# Omitted max_tokens. llama-server omits n_predict when this is not positive,
# so the model's own stop is the cap. 128 is not that cap.
UNLIMITED_N_PREDICT = -1


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value > 0:
        return int(value)
    return None


def completion_cap(max_tokens: Any, num_predict: Any) -> int:
    """OpenAI max_tokens, else options.num_predict, else unlimited."""
    n = _positive_int(max_tokens)
    if n is not None:
        return n
    n = _positive_int(num_predict)
    if n is not None:
        return n
    return UNLIMITED_N_PREDICT


def think_forces_legacy(think: Any) -> bool:
    """think:false stays on the runtime. Any other think value needs ggml."""
    return think is not None and think is not False


def loaded_ctx_must_grow(
    proc_alive: bool,
    same_path: bool,
    needed: int | None,
    loaded: int | None,
) -> bool:
    """Reload when the request needs a larger context than the one already loaded.

    An unknown loaded size is a warm start (LLAMA_MODEL already running). Growing
    then would stop and restart that process on the first request.
    """
    return bool(
        proc_alive
        and same_path
        and needed is not None
        and loaded is not None
        and needed > loaded
    )


def shared_completion_cap(caps: list[int]) -> int:
    """One cap for a mixed batch. A single unlimited row keeps the batch unlimited.

    Taking the max of a positive cap and -1 would clamp the unlimited row.
    """
    if not caps or any(c <= 0 for c in caps):
        return UNLIMITED_N_PREDICT
    return max(caps)


def admit_token_budget(n_predict: Any) -> int:
    """Scheduler reserve for a completion cap. Unlimited reserves no extra tokens.

    num_ctx still wins in the slot math when it is set. A negative budget is
    rejected by decode plans, so it must not be stored on the request.
    """
    if isinstance(n_predict, int) and n_predict > 0:
        return n_predict
    return 0


# Listeners the runtime must not take. 8081 is the sidecar itself, so the
# embed port may be 8081. llama-server binds embed+1 and must not land on
# the sidecar, the Go daemon, or storage.
_RESERVED_EMBED_PORTS = frozenset({11434, 18090})
_RESERVED_LLAMA_PORTS = frozenset({11434, 8081, 18090})


def runtime_port_conflict(embed_port: int) -> str | None:
    """None when embed_port and embed_port+1 stay off reserved listeners."""
    try:
        port = int(embed_port)
    except (TypeError, ValueError):
        return "runtime port is not an integer"
    llama = port + 1
    if port in _RESERVED_EMBED_PORTS:
        return f"runtime port {port} is reserved"
    if llama in _RESERVED_LLAMA_PORTS:
        return f"llama-server port {llama} collides with a reserved listener"
    return None


def apply_tool_choice(
    tools: list[dict[str, Any]], choice: Any
) -> list[dict[str, Any]]:
    """none drops tools. A named function keeps that tool. auto keeps the list."""
    if not tools or choice is None:
        return tools
    if isinstance(choice, str):
        name = choice.strip().lower()
        if name in ("", "auto", "required", "any"):
            return tools
        if name == "none":
            return []
        raise ValueError(f"unsupported tool_choice {choice!r}")
    if isinstance(choice, dict):
        fn = choice.get("function")
        if not isinstance(fn, dict):
            return tools
        want = str(fn.get("name") or "").strip()
        if not want:
            return tools
        kept = [
            t
            for t in tools
            if isinstance(t.get("function"), dict) and t["function"].get("name") == want
        ]
        if not kept:
            raise ValueError(f"tool_choice names unknown function {want!r}")
        return kept
    return tools
