#!/usr/bin/env bash
# Shared Darwin sidecar runtime + Go proxy startup (source only).
#
# Why this library exists: m3_metal_signoff.sh and serve_mac_runtime.sh share the same
# sidecar+Go bootstrap; duplicating curl loops and log redirection caused silent hangs.
#
#   source ./scripts/runtime/macos_runtime_serve_lib.sh
#   macos_runtime_urls
#   macos_runtime_start_sidecar "$LLAMA_MODEL"   # optional config path as 2nd arg
#   macos_runtime_start_go
#
# Env: OLLAMA_HOST, ZEROLLAMA_RUNTIME_URL, ZEROLLAMA_BIN, LLAMA_MODEL, LLAMA_CPP_*
set -euo pipefail

_MACOS_RT_ROOT="${MACOS_RT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=scripts/runtime/runtime_uv_venv.sh
source "${_MACOS_RT_ROOT}/scripts/runtime/runtime_uv_venv.sh"
# shellcheck source=scripts/runtime/runtime_smoke_lib.sh
source "${_MACOS_RT_ROOT}/scripts/runtime/runtime_smoke_lib.sh"

_MACOS_RT_PID=""
_MACOS_GO_PID=""
_MACOS_RT_HOST=""
_MACOS_RT_PORT=""

macos_runtime_log_paths() {
  export MACOS_RT_LOG="${MACOS_RT_LOG:-/tmp/macos-runtime.log}"
  export MACOS_GO_LOG="${MACOS_GO_LOG:-/tmp/macos-go.log}"
}

_macos_wait_http() {
  local label="$1"
  local url="$2"
  local max="${3:-30}"
  local curl_max="${4:-2}"
  local i
  echo -n "waiting for ${label} (${url})"
  for ((i = 1; i <= max; i++)); do
    if curl -sf -m "${curl_max}" "${url}" >/dev/null 2>&1; then
      echo " ok"
      return 0
    fi
    echo -n "."
    sleep 1
  done
  echo " failed"
  return 1
}

macos_runtime_print_ready() {
  macos_runtime_log_paths
  macos_runtime_urls
  cat <<EOF

=== macOS runtime stack ready ===
  Go API:      ${OLLAMA_HOST}
  Runtime:     ${ZEROLLAMA_RUNTIME_URL}
  Go log:      ${MACOS_GO_LOG}
  Runtime log: ${MACOS_RT_LOG}

Follow logs:
  tail -f "${MACOS_RT_LOG}" "${MACOS_GO_LOG}"

EOF
}

macos_runtime_urls() {
  # CI/sign-off layout: Go :8080 + sidecar :8081. Daily `zerollama serve` uses :11434 — do not
  # assume these defaults when curling a default serve; set OLLAMA_HOST explicitly in smokes.
  export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:8080}"
  export ZEROLLAMA_RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"
  _MACOS_RT_HOST="$(runtime_url_host "${ZEROLLAMA_RUNTIME_URL}")"
  _MACOS_RT_PORT="$(runtime_url_port "${ZEROLLAMA_RUNTIME_URL}" 8081)"
}

macos_resolve_llama_cpp_root() {
  if [[ -n "${LLAMA_CPP_ROOT:-}" ]]; then
    echo "${LLAMA_CPP_ROOT}"
    return 0
  fi
  local pin
  pin="$(grep '^FETCH_HEAD=' "${_MACOS_RT_ROOT}/Makefile.sync" 2>/dev/null | cut -d= -f2 || echo c84b3020)"
  local vendor="${_MACOS_RT_ROOT}/vendor/llama-cpp-${pin}"
  if [[ -f "${vendor}/CMakeLists.txt" && -f "${vendor}/build/bin/libllama.dylib" ]]; then
    echo "${vendor}"
    return 0
  fi
  echo "${_MACOS_RT_ROOT}/../llama.cpp"
}

macos_export_llama_cpp_paths() {
  export LLAMA_CPP_ROOT="$(macos_resolve_llama_cpp_root)"
  if [[ -f "${LLAMA_CPP_ROOT}/build/bin/libllama.dylib" ]]; then
    export LLAMA_CPP_LIB="${LLAMA_CPP_LIB:-${LLAMA_CPP_ROOT}/build/bin/libllama.dylib}"
  elif [[ -f "${LLAMA_CPP_ROOT}/build/bin/libllama.so" ]]; then
    export LLAMA_CPP_LIB="${LLAMA_CPP_LIB:-${LLAMA_CPP_ROOT}/build/bin/libllama.so}"
  fi
  if [[ -z "${LLAMA_SERVER_BIN:-}" && -x "${LLAMA_CPP_ROOT}/build/bin/llama-server" ]]; then
    export LLAMA_SERVER_BIN="${LLAMA_CPP_ROOT}/build/bin/llama-server"
  fi
}

macos_runtime_sidecar_cleanup() {
  [[ -n "${_MACOS_RT_PID}" ]] && kill "${_MACOS_RT_PID}" 2>/dev/null || true
  [[ -n "${_MACOS_GO_PID}" ]] && kill "${_MACOS_GO_PID}" 2>/dev/null || true
}

macos_runtime_stop_sidecar_port() {
  macos_runtime_urls
  # WHY also free port+1: subprocess llama-server binds runtime_port+1 (lab 18082).
  # Killing only the Python sidecar orphans llama-server and breaks the next L2 leg.
  lsof -ti :"${_MACOS_RT_PORT}" 2>/dev/null | xargs kill 2>/dev/null || true
  local llama_port=$((_MACOS_RT_PORT + 1))
  lsof -ti :"${llama_port}" 2>/dev/null | xargs kill 2>/dev/null || true
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
  # Preserve explicit backend override (L2/L3 smokes force subprocess for llama-server slots).
  if [[ -z "${ZEROLLAMA_RUNTIME_LLAMA_BACKEND:-}" ]]; then
    unset ZEROLLAMA_RUNTIME_LLAMA_BACKEND
  fi

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
  macos_export_llama_cpp_paths
  [[ -n "${require_model}" ]] && export LLAMA_MODEL="${require_model}"
  macos_runtime_log_paths
  echo "starting Python runtime sidecar on ${ZEROLLAMA_RUNTIME_URL} (log: ${MACOS_RT_LOG})"
  "${RUNTIME_UV_PYTHON}" -m runtime serve --host "${_MACOS_RT_HOST}" --port "${_MACOS_RT_PORT}" \
    >"${MACOS_RT_LOG}" 2>&1 &
  _MACOS_RT_PID=$!

  if ! _macos_wait_http "runtime /health" "${ZEROLLAMA_RUNTIME_URL%/}/health" "${MACOS_RT_HEALTH_MAX:-30}"; then
    tail -20 "${MACOS_RT_LOG}" >&2
    echo "runtime failed to start on ${ZEROLLAMA_RUNTIME_URL}" >&2
    return 1
  fi
  if [[ "$assert_m3" == "1" ]]; then
    smoke_runtime_assert_m3_inprocess_config "${ZEROLLAMA_RUNTIME_URL}"
  fi
  echo "runtime listening on ${ZEROLLAMA_RUNTIME_URL}"
  return 0
}

macos_runtime_training_env() {
  [[ "${OLLAMA_TRAINING:-true}" == "false" ]] && return 0
  # shellcheck source=scripts/training/training_uv_venv.sh
  source "${_MACOS_RT_ROOT}/scripts/training/training_uv_venv.sh"
  if [[ -x "${TRAINING_UV_VENV}/bin/python" ]] || [[ "${TRAINING_UV_AUTO:-0}" == "1" ]]; then
    training_uv_venv
    return 0
  fi
  echo "hint: run ./scripts/training/training_uv_venv.sh --verify for /api/train (sets PYTHONPATH via uv)" >&2
}

macos_runtime_start_go() {
  macos_runtime_urls
  local bin
  bin="$(smoke_resolve_zerollama_bin "${_MACOS_RT_ROOT}")"

  if curl -sf -m 15 "${OLLAMA_HOST%/}/api/tags" >/dev/null 2>&1; then
    echo "go api already listening on ${OLLAMA_HOST}"
    return 0
  fi
  macos_runtime_training_env
  export ZEROLLAMA_RUNTIME=1 OLLAMA_NO_CLOUD="${OLLAMA_NO_CLOUD:-true}"
  export ZEROLLAMA_RUNTIME_URL
  macos_runtime_log_paths
  echo "starting zerollama serve on ${OLLAMA_HOST} (log: ${MACOS_GO_LOG})"
  "${bin}" serve >"${MACOS_GO_LOG}" 2>&1 &
  _MACOS_GO_PID=$!
  if ! _macos_wait_http "zerollama /api/tags" "${OLLAMA_HOST%/}/api/tags" "${MACOS_GO_TAGS_MAX:-30}" 15; then
    tail -20 "${MACOS_GO_LOG}" >&2
    echo "zerollama serve failed to start (${bin})" >&2
    return 1
  fi
  echo "zerollama listening on ${OLLAMA_HOST}"
  return 0
}
