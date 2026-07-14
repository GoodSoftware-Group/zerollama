#!/usr/bin/env bash
# Print effective runtime env (L3, KV, autoconfig) without a live server.
#
# WHY: operator shells accumulate stale ZEROLLAMA_* exports; this shows what
# the sidecar would actually use after YAML/profile/smart defaults.
#
# Usage:
#   ./scripts/runtime/runtime_env_doctor.sh
#   ZEROLLAMA_L3_PROFILE=agent ./scripts/runtime/runtime_env_doctor.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}/runtime"

PYTHONPATH=. python3 <<'PY'
import json

from runtime.autoconfig import autoconfig_health, resolved_config_path
from runtime.env import runtime_env_health
from runtime.llama_cpp_unified import (
    normalize_llama_cpp_env,
    resolve_llama_cpp_root,
    resolve_llama_cpp_lib,
    resolve_llama_server_bin,
)
from runtime.llama_patch_health import llama_patch_health

root = resolve_llama_cpp_root()
report = {
    "resolved_config": str(resolved_config_path()),
    "autoconfig": autoconfig_health(),
    "llama_cpp_root": str(root),
    "llama_server_bin": str(resolve_llama_server_bin(root) or ""),
    "llama_cpp_lib": str(resolve_llama_cpp_lib(root) or ""),
    "llama_cpp_notes": normalize_llama_cpp_env(),
    "llama_patches": llama_patch_health(),
    "runtime_env": runtime_env_health(),
}
print(json.dumps(report, indent=2))
PY
