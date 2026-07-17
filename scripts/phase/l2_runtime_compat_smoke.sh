#!/usr/bin/env bash
# L2 runtime subprocess compat — L1 vs fork profiles on unified llama-server.
#
# WHY: qwen35 Go ggml smoke does not exercise llama-server. This validates the runtime
# subprocess path loads GGUF and decodes with stock cache types and fork (QJL/TBQ) argv.
#
# Usage:
#   M3_LLAMA_MODEL=/path/to/model.gguf ./scripts/phase/l2_runtime_compat_smoke.sh
#   L2_BUILD_FORK=1 ./scripts/phase/l2_runtime_compat_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/runtime/runtime_uv_venv.sh
source "${ROOT}/scripts/runtime/runtime_uv_venv.sh"
# shellcheck source=scripts/runtime/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime/runtime_smoke_lib.sh"
# shellcheck source=scripts/runtime/macos_runtime_serve_lib.sh
source "${ROOT}/scripts/runtime/macos_runtime_serve_lib.sh"

runtime_uv_venv

UNIFIED_ROOT="${LLAMA_CPP_ROOT:-$(cd "${ROOT}/.." && pwd)/llama.cpp}"
STOCK_ROOT="${STOCK_LLAMA_CPP_ROOT:-${UNIFIED_ROOT}}"
FORK_ROOT="${ELIZA_LLAMA_CPP_ROOT:-${UNIFIED_ROOT}}"

if [[ "${L2_BUILD:-0}" == "1" || "${L2_BUILD_FORK:-0}" == "1" ]]; then
  LLAMA_CPP_ROOT="${UNIFIED_ROOT}" "${ROOT}/scripts/build/build_llama_server.sh"
fi

smoke_m3_resolve_signoff_model

export ZEROLLAMA_GPU_PROFILE=1
export ZEROLLAMA_RUNTIME_LLAMA_BACKEND=subprocess
unset ZEROLLAMA_RUNTIME_CONFIG
export ZEROLLAMA_AUTO_CONFIG=1
export MACOS_RT_HEALTH_MAX="${MACOS_RT_HEALTH_MAX:-120}"

macos_runtime_urls
trap macos_runtime_sidecar_cleanup EXIT

_l2_compat_leg() {
  local label="$1"
  local cpp_root="$2"
  local fork_mode="$3"

  echo ""
  echo "== L2 runtime compat: ${label} (${cpp_root}) =="

  export LLAMA_CPP_ROOT="${cpp_root}"
  export LLAMA_SERVER_BIN="${cpp_root}/build/bin/llama-server"
  export LLAMA_CPP_LIB="${cpp_root}/build/bin/libllama.dylib"
  export LLAMA_MODEL

  case "${fork_mode}" in
    off) export ZEROLLAMA_LLAMA_FORK=0 ;;
    auto) unset ZEROLLAMA_LLAMA_FORK ;;
    *) echo "unknown fork_mode: ${fork_mode}" >&2; return 1 ;;
  esac

  if [[ ! -x "${LLAMA_SERVER_BIN}" ]]; then
    echo "missing ${LLAMA_SERVER_BIN}" >&2
    return 1
  fi

  macos_runtime_stop_sidecar_port
  macos_runtime_start_sidecar "${LLAMA_MODEL}" "" 0

  local health_json
  health_json="$(runtime_fetch_health "${ZEROLLAMA_RUNTIME_URL}")"
  runtime_resume_if_needed "${health_json}"

  export ZEROLLAMA_RUNTIME_URL
  python3 <<'PY'
import json
import os
import urllib.request

url = os.environ["ZEROLLAMA_RUNTIME_URL"].rstrip("/")
gguf = os.environ["LLAMA_MODEL"]
payload = {
    "model": "l2-compat",
    "prompt": "Reply with exactly: ok",
    "stream": False,
    "options": {"gguf": gguf, "num_ctx": 4096, "num_predict": 8},
}
req = urllib.request.Request(
    f"{url}/api/generate",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=600) as resp:
    out = json.loads(resp.read().decode())
assert out.get("done"), out
text = (out.get("response") or "").strip()
assert text, out
health = json.loads(
    urllib.request.urlopen(f"{url}/health", timeout=30).read().decode()
)
lf = health.get("llama_fork") or {}
gp = health.get("gpu_profile") or {}
print(f"generate ok: {text[:60]!r}")
print(f"llama_fork.enabled={lf.get('enabled')} source={lf.get('source')}")
print(f"gpu_profile.llama_fork={gp.get('llama_fork')}")
args = gp.get("llama_server_args") or health.get("llama_server_args")
if isinstance(args, list):
    for i, a in enumerate(args):
        if a == "--cache-type-k" and i + 1 < len(args):
            print(f"cache_type_k={args[i+1]}")
PY

  macos_runtime_stop_sidecar_port
}

_l2_compat_leg "stock" "${STOCK_ROOT}" "off"
_l2_compat_leg "fork" "${FORK_ROOT}" "auto"

trap - EXIT
echo ""
echo "PASS: l2_runtime_compat_smoke"
echo "Doc: docs/gpu-profiles-l2.md"
