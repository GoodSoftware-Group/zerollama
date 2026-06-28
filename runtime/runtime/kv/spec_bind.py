"""KVCacheSpec ↔ Phase 15 page bind (vLLM retention × PA validation).

WHY: page_bind validates token positions against the PA block table; KVCacheSpec
validates whether prefix *reuse* is semantically safe (SWA window, draft spec).
This module joins both checks at decode admission so operators see one failure
surface instead of silent wrong KV.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from runtime.kv_cache_spec import KVCacheSpec, PrefixCacheRequest
from runtime.worker.llama_server import LlamaServerError

if TYPE_CHECKING:
    from runtime.scheduler.scheduler import Request


def prefix_within_spec(
    spec: KVCacheSpec,
    *,
    seq_pos: int | None = None,
    prompt_tokens: int | None = None,
) -> bool:
    """True when spec allows prefix retention for this seq_pos + prompt size."""
    # SWA checks ignore cache key; "probe" satisfies PrefixCacheRequest shape only.
    req = PrefixCacheRequest(
        prompt_cache_key="probe",
        seq_pos=seq_pos,
        prompt_tokens=prompt_tokens,
    )
    return spec.swa_allows_cache_prompt(
        seq_pos=req.seq_pos, prompt_tokens=req.prompt_tokens
    )


def assert_prefix_within_spec(
    spec: KVCacheSpec,
    *,
    seq_pos: int | None,
    prompt_tokens: int | None,
    at: str = "decode",
) -> None:
    """Fail fast when SWA window would make prefix reuse invalid."""
    if prefix_within_spec(spec, seq_pos=seq_pos, prompt_tokens=prompt_tokens):
        return
    window = spec.effective_window
    raise LlamaServerError(
        f"prefix cache ({at}): seq_pos={seq_pos or 0} + "
        f"prompt_tokens={prompt_tokens or 0} exceeds SWA effective_window={window}"
    )


def validate_decode_prefix(
    spec: KVCacheSpec,
    req: Request,
    *,
    decode_pos: int,
    n_prompt: int,
    cache_prompt: bool | None,
    at: str = "decode",
) -> None:
    """Validate PA decode range + spec retention when cache_prompt is requested."""
    if cache_prompt is False:
        return
    if not req.prompt_cache_key:
        return
    if not spec.allow_cache_prompt_base:
        return
    assert_prefix_within_spec(
        spec,
        seq_pos=decode_pos,
        prompt_tokens=n_prompt,
        at=at,
    )


def spec_bind_health(spec: KVCacheSpec) -> dict[str, Any]:
    """Operator hints linking L3 spec to Phase 15 bind."""
    return {
        "kind": spec.kind,
        "effective_window": spec.effective_window,
        "swa_enforced": spec.kind in ("sliding_window", "hybrid")
        and spec.effective_window is not None,
        "hybrid_coordinator": (
            spec.coordinator.to_health() if spec.coordinator is not None else None
        ),
        "draft_spec_drop_last_block": spec.drop_last_block_on_resume,
        "swa_retention_interval": spec.retention_interval,
    }


def resume_allowed_by_spec(
    spec: KVCacheSpec,
    *,
    prompt_cache_key: str | None,
    seq_pos: int,
    n_prompt: int,
    cache_prompt: bool | None,
) -> bool:
    """Whether in-process KV resume is safe for this pinned turn."""
    if cache_prompt is False:
        return False
    if not prompt_cache_key:
        return True
    if not spec.allow_cache_prompt_base:
        return False
    return prefix_within_spec(spec, seq_pos=seq_pos, prompt_tokens=n_prompt)
