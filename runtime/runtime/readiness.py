"""Runtime readiness (orchestrator / systemd probes).

``/health`` stays HTTP 200 with rich diagnostics. ``/ready`` and the ``ready``
field fail when inference cannot accept new work.
"""

from __future__ import annotations

from typing import Any


def compute_readiness(body: dict[str, Any]) -> dict[str, Any]:
    """Derive ``ready`` + reasons from a built ``/health`` body."""
    reasons: list[str] = []
    warnings: list[str] = []

    if not body.get("accepts_new_loads"):
        state = body.get("inference_state", "?")
        reasons.append(f"not accepting loads (inference_state={state})")

    fork = body.get("llama_fork") or {}
    if fork.get("enabled"):
        if fork.get("source") == "probe_disabled_cuda_backend":
            reasons.append("fork KV enabled but CUDA backend lacks SET_ROWS/fused attn")
        elif fork.get("cuda_backend_capable") is False:
            reasons.append("fork KV enabled but cuda_backend_capable=false")

    unified = body.get("llama_cpp_unified") or {}
    needs_llama = bool(unified.get("llama_server_bin") or body.get("llama_model"))
    if unified.get("runtime_ready") is False and needs_llama:
        reasons.append("llama_cpp_unified.runtime_ready=false")

    server_bin = unified.get("llama_server_bin")
    backend = body.get("llama_backend", "")
    if (
        backend == "subprocess"
        and not server_bin
        and body.get("llama_model")
    ):
        reasons.append("subprocess backend but LLAMA_SERVER_BIN unset")

    status = body.get("llama_server_status") or {}
    if status.get("died"):
        reasons.append(
            f"llama-server exited (code={status.get('exit_code')})"
        )
    elif status.get("running") and status.get("reachable") is False:
        reasons.append("llama-server running but /health unreachable")

    patches = body.get("llama_patches") or {}
    if patches.get("status") == "fail":
        if patches.get("deployment_mode") == "external_binary":
            if patches.get("llama_server_binary_seq_copy") is False:
                reasons.append("external llama-server lacks /kv/seq-copy")
            else:
                warnings.append("in-tree patch markers skipped (external binary install)")
        else:
            for issue in (patches.get("issues") or [])[:3]:
                warnings.append(f"llama_patches: {issue}")

    admission = body.get("admission") or {}
    if admission.get("training_handoff_active") and not body.get("accepts_new_loads"):
        warnings.append("training_handoff_active (awaiting resume or auto-resume)")

    ready = len(reasons) == 0
    return {
        "ready": ready,
        "ready_reasons": reasons,
        "ready_warnings": warnings,
    }
