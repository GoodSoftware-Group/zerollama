#!/usr/bin/env bash
# Shared Linux/CUDA sidecar runtime startup (source only).
#
# WHY this library exists: the Mac path uses lsof + .dylib + apple_silicon.yaml;
# Linux/CUDA needs fuser + .so + single_gpu.yaml.  Mirrors macos_runtime_serve_lib.sh
# so L2 and other scripts can source one or the other based on uname.
#
#   source ./scripts/linux_runtime_serve_lib.sh
#   linux_runtime_urls
#   linux_runtime_start_sidecar "$LLAMA_MODEL"
#   # ... bench ...
#   linux_runtime_stop_sidecar_port
#
# Env: OLLAMA_HOST, ZEROLLAMA_RUNTIME_URL, LLAMA_MODEL, LLAMA_CPP_*
#      LINUX_RT_HEALTH_MAX, LINUX_RT_CURL_TIMEOUT (default 15 — cold /health can take ~9s on CUDA)
# shellcheck shell=bash

_LINUX_RT_ROOT="${LINUX_RT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=scripts/runtime_uv_venv.sh
source "${_LINUX_RT_ROOT}/scripts/runtime_uv_venv.sh"
# shellcheck source=scripts/runtime_smoke_lib.sh
source "${_LINUX_RT_ROOT}/scripts/runtime_smoke_lib.sh"

# WHY module-level PID/HOST/PORT vars instead of subshell: sourced libraries
# cannot export state back to the caller via subshells.  These globals let
# linux_runtime_sidecar_cleanup() kill the background process even after the
# caller's trap fires, matching the macos_runtime_serve_lib.sh pattern.
_LINUX_RT_PID=""
_LINUX_RT_HOST=""
_LINUX_RT_PORT=""

linux_runtime_log_paths() {
  export LINUX_RT_LOG="${LINUX_RT_LOG:-/tmp/linux-runtime.log}"
}

_linux_wait_http() {
  local label="$1"
  local url="$2"
  local max="${3:-30}"
  local curl_to="${LINUX_RT_CURL_TIMEOUT:-15}"
  local i
  echo -n "waiting for ${label} (${url})"
  for ((i = 1; i <= max; i++)); do
    if curl -sf -m "${curl_to}" "${url}" >/dev/null 2>&1; then
      echo " ok"
      return 0
    fi
    echo -n "."
    sleep 1
  done
  echo " failed"
  return 1
}

linux_runtime_urls() {
  export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:8080}"
  export ZEROLLAMA_RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"
  _LINUX_RT_HOST="$(runtime_url_host "${ZEROLLAMA_RUNTIME_URL}")"
  _LINUX_RT_PORT="$(runtime_url_port "${ZEROLLAMA_RUNTIME_URL}" 8081)"
}

linux_runtime_sidecar_cleanup() {
  [[ -n "${_LINUX_RT_PID:-}" ]] && kill "${_LINUX_RT_PID}" 2>/dev/null || true
}

linux_runtime_stop_sidecar_port() {
  linux_runtime_urls
  # WHY fuser, not lsof: lsof is an optional package on most Linux distros;
  # fuser is part of util-linux and always present on CI runners.
  # macOS equivalent uses lsof (see macos_runtime_serve_lib.sh).
  fuser -k "${_LINUX_RT_PORT}/tcp" 2>/dev/null || true
  # WHY port+1: subprocess llama-server binds loopback at runtime_port+1.
  fuser -k "$((_LINUX_RT_PORT + 1))/tcp" 2>/dev/null || true
  sleep 1
}

# Start uv sidecar on Linux/CUDA.
# Args: require_model path, optional ZEROLLAMA_RUNTIME_CONFIG
linux_runtime_start_sidecar() {
  local require_model="${1:-}"
  local config="${2:-}"
  linux_runtime_urls

  if [[ -n "$config" ]]; then
    export ZEROLLAMA_RUNTIME_CONFIG="$config"
    unset ZEROLLAMA_AUTO_CONFIG
  else
    unset ZEROLLAMA_RUNTIME_CONFIG
    export ZEROLLAMA_AUTO_CONFIG="${ZEROLLAMA_AUTO_CONFIG:-1}"
  fi
  if [[ -z "${ZEROLLAMA_RUNTIME_LLAMA_BACKEND:-}" ]]; then
    unset ZEROLLAMA_RUNTIME_LLAMA_BACKEND
  fi

  # Restart when config differs or model changed.
  if [[ -n "$config" ]]; then
    if curl -sf -m 2 "${ZEROLLAMA_RUNTIME_URL%/}/health" >/dev/null 2>&1; then
      echo "restarting runtime (explicit config: ${config})"
      linux_runtime_stop_sidecar_port
    fi
  elif curl -sf -m 2 "${ZEROLLAMA_RUNTIME_URL%/}/health" >/dev/null 2>&1; then
    local have=""
    have="$(curl -sf -m 2 "${ZEROLLAMA_RUNTIME_URL%/}/health" | python3 -c \
      'import json,sys; print(json.load(sys.stdin).get("llama_model") or "")' 2>/dev/null || true)"
    if [[ -z "${require_model}" || "${have}" == "${require_model}" ]]; then
      echo "runtime already listening on ${ZEROLLAMA_RUNTIME_URL} (model ok)"
      return 0
    fi
    echo "restarting runtime (llama_model mismatch)"
    linux_runtime_stop_sidecar_port
  fi

  runtime_uv_venv
  # WHY default LLAMA_CPP_ROOT to ../llama.cpp (sibling of zerollama repo):
  # the fork sits at ../eliza-llama.cpp; stock at ../llama.cpp.  Callers
  # override via STOCK_LLAMA_CPP_ROOT / ELIZA_LLAMA_CPP_ROOT env when running
  # A/B bench legs — this default covers the "just start a runtime" use case.
  export LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-${_LINUX_RT_ROOT}/../llama.cpp}"
  [[ -n "${require_model}" ]] && export LLAMA_MODEL="${require_model}"
  linux_runtime_log_paths
  echo "starting Python runtime sidecar on ${ZEROLLAMA_RUNTIME_URL} (log: ${LINUX_RT_LOG})"
  "${RUNTIME_UV_PYTHON}" -m runtime serve \
    --host "${_LINUX_RT_HOST}" --port "${_LINUX_RT_PORT}" \
    >"${LINUX_RT_LOG}" 2>&1 &
  _LINUX_RT_PID=$!

  local max="${LINUX_RT_HEALTH_MAX:-120}"
  if ! _linux_wait_http "runtime /health" "${ZEROLLAMA_RUNTIME_URL%/}/health" "${max}"; then
    tail -20 "${LINUX_RT_LOG}" >&2
    echo "runtime failed to start on ${ZEROLLAMA_RUNTIME_URL}" >&2
    return 1
  fi
}

linux_runtime_resume_if_needed() {
  local health_json="${1:-}"
  if [[ -z "${health_json}" ]]; then
    health_json="$(curl -sf "${ZEROLLAMA_RUNTIME_URL%/}/health" || true)"
  fi
  local paused
  paused="$(echo "${health_json}" | python3 -c \
    'import json,sys; d=json.load(sys.stdin); print(d.get("ggml_paused") or "")' 2>/dev/null || true)"
  if [[ "${paused}" == "true" ]]; then
    echo "resuming paused ggml"
    curl -sf -X POST "${ZEROLLAMA_RUNTIME_URL%/}/internal/ggml-resume" >/dev/null || true
    sleep 1
  fi
}
