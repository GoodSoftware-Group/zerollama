#!/usr/bin/env bash
# Print effective runtime env (L3, KV, autoconfig) without a live server.
#
# WHY: operator shells accumulate stale ZEROLLAMA_* exports; this shows what
# the sidecar would actually use after YAML/profile/smart defaults.
#
# Usage:
#   ./scripts/runtime/runtime_env_doctor.sh
#   ZEROLLAMA_L3_PROFILE=agent ./scripts/runtime/runtime_env_doctor.sh
#   ZEROLLAMA_INFERENCE_PROFILE=agent ./scripts/runtime/runtime_env_doctor.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}/runtime"

PYTHONPATH=. python3 <<'PY'
import json
import os

from runtime.autoconfig import autoconfig_health, resolved_config_path
from runtime.env import runtime_env_health
from runtime.llama_cpp_unified import (
    normalize_llama_cpp_env,
    resolve_llama_cpp_root,
    resolve_llama_cpp_lib,
    resolve_llama_server_bin,
)
from runtime.llama_patch_health import llama_patch_health

def _inference_profile_note() -> dict:
    """Mirror Go envconfig.ApplyInferenceProfileDefaults for operator visibility.

    Go serve applies this at startup; the Python sidecar inherits the process env.
    """
    raw = (os.environ.get("ZEROLLAMA_INFERENCE_PROFILE") or "").strip().lower()
    requested = raw or "(default→auto)"
    resolved = {
        "": "throughput",
        "auto": "throughput",
        "throughput": "throughput",
        "agent": "agent",
        "vram": "vram",
        "off": "off",
        "0": "off",
        "false": "off",
        "no": "off",
    }.get(raw, "throughput")
    soft = {
        "ZEROLLAMA_GPU_PROFILE": "1",
        "ZEROLLAMA_LLAMA_FORK": "0",
        "ZEROLLAMA_LLAMA_CACHE": "1",
        "GGML_CUDA_USE_GRAPHS": "0",
    }
    if resolved == "agent":
        soft["ZEROLLAMA_L3_PROFILE"] = "agent"
        soft["ZEROLLAMA_RADIX_PREFIX_SHARE"] = "1"
    elif resolved == "vram":
        soft["ZEROLLAMA_LLAMA_FORK_AUTO_VRAM"] = "1"
        soft["ZEROLLAMA_LLAMA_FORK_PROFILE"] = "vram"
    would_apply = []
    if resolved != "off":
        for k, v in soft.items():
            if not (os.environ.get(k) or "").strip():
                would_apply.append(f"{k}={v}")
    return {
        "requested": requested,
        "resolved": resolved,
        "would_soft_set_if_unset": would_apply,
        "doc": "docs/gpu-profiles-l1.md — prefer INFERENCE_PROFILE over flag soup",
    }

root = resolve_llama_cpp_root()
report = {
    "resolved_config": str(resolved_config_path()),
    "autoconfig": autoconfig_health(),
    "inference_profile": _inference_profile_note(),
    "llama_cpp_root": str(root),
    "llama_server_bin": str(resolve_llama_server_bin(root) or ""),
    "llama_cpp_lib": str(resolve_llama_cpp_lib(root) or ""),
    "llama_cpp_notes": normalize_llama_cpp_env(),
    "llama_patches": llama_patch_health(),
    "runtime_env": runtime_env_health(),
}
print(json.dumps(report, indent=2))
PY
