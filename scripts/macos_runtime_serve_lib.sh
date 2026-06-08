#!/usr/bin/env bash
# Shared Darwin sidecar runtime + Go proxy startup (source only).
#
#   source ./scripts/macos_runtime_serve_lib.sh
#   macos_runtime_urls
#   macos_runtime_start_sidecar "$LLAMA_MODEL"   # optional config path as 2nd arg
#   macos_runtime_start_go
#
# Env: OLLAMA_HOST, ZEROLLAMA_RUNTIME_URL, ZEROLLAMA_BIN, LLAMA_MODEL, LLAMA_CPP_*
set -euo pipefail

_MACOS_RT_ROOT="${MACOS_RT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=scripts/runtime_uv_venv.sh
source "${_MACOS_RT_ROOT}/scripts/runtime_uv_venv.sh"
# shellcheck source=scripts/runtime_smoke_lib.sh
source "${_MACOS_RT_ROOT}/scripts/runtime_smoke_lib.sh"

_MACOS_RT_PID=""
_MACOS_GO_PID=""
_MACOS_RT_HOST=""
_MACOS_RT_PORT=""

macos_runtime_urls() {
  export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:8080}"
  export ZEROLLAMA_RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"
  _MACOS_RT_HOST="$(runtime_url_host "${ZEROLLAMA_RUNTIME_URL}")"
  _MACOS_RT_PORT="$(runtime_url_port "${ZEROLLAMA_RUNTIME_URL}" 8081)"
}

macos_runtime_sidecar_cleanup() {
  [[ -n "${_MACOS_RT_PID}" ]] && kill "${_MACOS_RT_PID}" 2>/dev/null || true
  [[ -n "${_MACOS_GO_PID}" ]] && kill "${_MACOS_GO_PID}" 2>/dev/null || true
}

macos_runtime_stop_sidecar_port() {
  macos_runtime_urls
  lsof -ti :"${_MACOS_RT_PORT}" 2>/dev/null | xargs kill 2>/dev/null || true
  sleep 1
}

# Start uv sidecar. Args: require_model path, optional ZEROLLAMA_RUNTIME_CONFIG, assert_m3 (0|1).
macos_runtime_start_sidecar() {
  local require_model="${1:-}"
  local config="${2:-}"
  local assert_m3="${3:-0}"
  macos_runtime_urls

  if [[ -n "$config" ]]; then
    export ZEROLLAMA_RUNTIME_CONFIG="$config"
    unset ZEROLLAMA_AUTO_CONFIG
  else
    unset ZEROLLAMA_RUNTIME_CONFIG
    export ZEROLLAMA_AUTO_CONFIG="${ZEROLLAMA_AUTO_CONFIG:-1}"
  fi
  unset ZEROLLAMA_RUNTIME_LLAMA_BACKEND

  # Explicit YAML (e.g. multiseq) must reload — do not reuse a running sidecar on another config.
  if [[ -n "$config" ]]; then
    if curl -sf -m 2 "${ZEROLLAMA_RUNTIME_URL%/}/health" >/dev/null 2>&1; then
      echo "restarting runtime (explicit config: ${config})"
      macos_runtime_stop_sidecar_port
    fi
  elif curl -sf -m 2 "${ZEROLLAMA_RUNTIME_URL%/}/health" >/dev/null 2>&1; then
    local have=""
    have="$(curl -sf -m 2 "${ZEROLLAMA_RUNTIME_URL%/}/health" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("llama_model") or "")' 2>/dev/null || true)"
    if [[ -z "${require_model}" || "${have}" == "${require_model}" ]]; then
      if [[ "$assert_m3" == "1" ]]; then
        smoke_runtime_assert_m3_inprocess_config "${ZEROLLAMA_RUNTIME_URL}"
      fi
      echo "runtime already listening on ${ZEROLLAMA_RUNTIME_URL} (model ok)"
      return 0
    fi
    echo "restarting runtime (llama_model mismatch)"
    macos_runtime_stop_sidecar_port
  fi

  runtime_uv_venv
  export LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-${_MACOS_RT_ROOT}/../llama.cpp}"
  [[ -n "${require_model}" ]] && export LLAMA_MODEL="${require_model}"
  "${RUNTIME_UV_PYTHON}" -m runtime serve --host "${_MACOS_RT_HOST}" --port "${_MACOS_RT_PORT}" \
    >"${MACOS_RT_LOG:-/tmp/macos-runtime.log}" 2>&1 &
  _MACOS_RT_PID=$!

  local health=""
  for _ in $(seq 1 30); do
    if health="$(curl -sf -m 2 "${ZEROLLAMA_RUNTIME_URL%/}/health" 2>/dev/null)"; then
      if [[ "$assert_m3" == "1" ]]; then
        smoke_runtime_assert_m3_inprocess_config "${ZEROLLAMA_RUNTIME_URL}"
      fi
      return 0
    fi
    sleep 1
  done
  tail -20 "${MACOS_RT_LOG:-/tmp/macos-runtime.log}" >&2
  echo "runtime failed to start on ${ZEROLLAMA_RUNTIME_URL}" >&2
  return 1
}

macos_runtime_start_go() {
  macos_runtime_urls
  local bin
  bin="$(smoke_resolve_zerollama_bin "${_MACOS_RT_ROOT}")"

  if curl -sf -m 2 "${OLLAMA_HOST%/}/api/tags" >/dev/null 2>&1; then
    echo "go api already listening on ${OLLAMA_HOST}"
    return 0
  fi
  export ZEROLLAMA_RUNTIME=1 OLLAMA_NO_CLOUD="${OLLAMA_NO_CLOUD:-true}"
  export ZEROLLAMA_RUNTIME_URL
  "${bin}" serve >"${MACOS_GO_LOG:-/tmp/macos-go.log}" 2>&1 &
  _MACOS_GO_PID=$!
  for _ in $(seq 1 20); do
    curl -sf -m 2 "${OLLAMA_HOST%/}/api/tags" >/dev/null && return 0
    sleep 1
  done
  tail -20 "${MACOS_GO_LOG:-/tmp/macos-go.log}" >&2
  echo "zerollama serve failed to start (${bin})" >&2
  return 1
}
