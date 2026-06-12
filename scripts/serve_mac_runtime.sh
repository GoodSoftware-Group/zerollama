#!/usr/bin/env bash
# Apple Silicon: uv sidecar runtime + zerollama Go proxy (CI / sign-off; daily use: zerollama serve).
#
// Why sidecar not embed: macOS system Python is often 3.9; runtime requires 3.10+.
// Why log files not stdout: CI/sign-off scripts need clean terminals; script prints paths + tail hint.
# apple_silicon.yaml sets llama_backend: inprocess when autoconfig picks darwin.
#
# Prerequisite:
#   LLAMA_CPP_ROOT=../llama.cpp ./scripts/build_llama_server.sh
#
# Usage:
#   export M3_LLAMA_MODEL=/path/to/text-only.gguf   # optional
#   ./scripts/serve_mac_runtime.sh
#
# Env:
#   ZEROLLAMA_RUNTIME_URL  — sidecar URL (default :8081); disables Go embed
#   OLLAMA_HOST            — Go API (default :8080)
#   ZEROLLAMA_BIN          — override zerollama binary (repo ./zerollama, then PATH)
#   MACOS_RT_LOG           — Python runtime log (default /tmp/macos-runtime.log)
#   MACOS_GO_LOG           — zerollama serve log (default /tmp/macos-go.log)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/macos_runtime_serve_lib.sh
source "${ROOT}/scripts/macos_runtime_serve_lib.sh"

LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-${ROOT}/../llama.cpp}"
export LLAMA_CPP_LIB="${LLAMA_CPP_LIB:-${LLAMA_CPP_ROOT}/build/bin/libllama.dylib}"
export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${LLAMA_CPP_ROOT}/build/bin/llama-server}"
export ZEROLLAMA_RUNTIME="${ZEROLLAMA_RUNTIME:-1}"
export ZEROLLAMA_AUTO_CONFIG="${ZEROLLAMA_AUTO_CONFIG:-1}"
export OLLAMA_NO_CLOUD="${OLLAMA_NO_CLOUD:-true}"
unset ZEROLLAMA_RUNTIME_LLAMA_BACKEND
unset ZEROLLAMA_RUNTIME_EMBED

if [[ ! -f "${LLAMA_CPP_LIB}" ]]; then
  echo "Missing ${LLAMA_CPP_LIB}; run: LLAMA_CPP_ROOT=../llama.cpp ./scripts/build_llama_server.sh" >&2
  exit 1
fi

macos_runtime_urls
_model="${M3_LLAMA_MODEL:-${LLAMA_MODEL:-}}"

_wait_pids=()
_cleanup() {
  macos_runtime_sidecar_cleanup
}
trap _cleanup EXIT INT TERM

if curl -sf -m 2 "${ZEROLLAMA_RUNTIME_URL%/}/health" >/dev/null 2>&1; then
  echo "runtime already on ${ZEROLLAMA_RUNTIME_URL}"
else
  macos_runtime_start_sidecar "${_model}" "" 1
  [[ -n "${_MACOS_RT_PID}" ]] && _wait_pids+=("${_MACOS_RT_PID}")
fi

if curl -sf -m 2 "${OLLAMA_HOST%/}/api/tags" >/dev/null 2>&1; then
  echo "zerollama already on ${OLLAMA_HOST}"
else
  macos_runtime_start_go
  [[ -n "${_MACOS_GO_PID}" ]] && _wait_pids+=("${_MACOS_GO_PID}")
fi

if ((${#_wait_pids[@]} == 0)); then
  echo "runtime and zerollama already running (${ZEROLLAMA_RUNTIME_URL}, ${OLLAMA_HOST})"
  macos_runtime_print_ready
  exit 0
fi

macos_runtime_print_ready
wait "${_wait_pids[@]}"
