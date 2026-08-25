#!/usr/bin/env bash
# Shared Linux/CUDA sidecar runtime startup (source only).
#
# WHY this library exists: the Mac path uses lsof + .dylib + apple_silicon.yaml;
# Linux/CUDA needs fuser + .so + single_gpu.yaml.  Mirrors macos_runtime_serve_lib.sh
# so L2 and other scripts can source one or the other based on uname.
#
#   source ./scripts/runtime/linux_runtime_serve_lib.sh
#   linux_runtime_urls
#   linux_runtime_start_sidecar "$LLAMA_MODEL"
#   # ... bench ...
#   linux_runtime_stop_sidecar_port
#
# Env: OLLAMA_HOST, ZEROLLAMA_RUNTIME_URL, LLAMA_MODEL, LLAMA_CPP_*
#      LINUX_RT_HEALTH_MAX, LINUX_RT_CURL_TIMEOUT (default 15 — cold /health can take ~9s on CUDA)
# shellcheck shell=bash

_LINUX_RT_ROOT="${LINUX_RT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=scripts/runtime/runtime_uv_venv.sh
source "${_LINUX_RT_ROOT}/scripts/runtime/runtime_uv_venv.sh"
# shellcheck source=scripts/runtime/runtime_smoke_lib.sh
source "${_LINUX_RT_ROOT}/scripts/runtime/runtime_smoke_lib.sh"

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
  # WHY not curl -m 2: InferenceEngine.health() on cold CUDA can take ~9s (nvidia probe +
  # scheduler snapshot). Shorter timeout made every attempt fail for 120s while uvicorn was up.
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
  # WHY port+1: subprocess llama-server binds loopback at runtime_port+1 (e.g. 8082).
  # Killing :8081 alone leaves orphan llama-server holding GPU VRAM across L1/L2 A/B legs.
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
  # WHY default LLAMA_CPP_ROOT to ../llama.cpp (unified sibling):
  # one llama-server binary; L2 bench legs differ by ZEROLLAMA_LLAMA_FORK only.
  export LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-${_LINUX_RT_ROOT}/../llama.cpp}"
  [[ -n "${require_model}" ]] && export LLAMA_MODEL="${require_model}"
  # WHY PYTHONPATH: package lives at runtime/runtime/; `python -m runtime` needs the
  # outer runtime/ on sys.path (systemd units set this; bare gate scripts did not).
  export PYTHONPATH="${_LINUX_RT_ROOT}/runtime${PYTHONPATH:+:${PYTHONPATH}}"
  # WHY CUDA rpath: vendor/llama-server often links libcudart.so.12 while the host
  # defaults to CUDA 13 — prefer bin dir + packaged cuda_v12 before system libs.
  linux_runtime_export_llama_ld_path
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

# Prepend llama-server bindir + CT GPU driver stubs + Ollama CUDA runtimes.
# WHY /root/nvidia-host first on Proxmox CT 1564: host libcuda via that path;
# without it, ctypes `llama_backend_init` and a second llama-server abort with
# `munmap_chunk(): invalid pointer` while the production process (started via
# serve_gpu_example.sh) keeps working. Match scripts/serve/serve_gpu_example.sh.
linux_runtime_export_llama_ld_path() {
  local bin="${LLAMA_SERVER_BIN:-}"
  local bin_dir=""
  local extras=()
  # Driver / CT stubs before CUDA runtimes (order matches production serve).
  [[ -d /root/nvidia-host ]] && extras+=("/root/nvidia-host")
  [[ -d /usr/hostlibs ]] && extras+=("/usr/hostlibs")
  if [[ -x "${bin}" ]]; then
    bin_dir="$(cd "$(dirname "${bin}")" && pwd)"
    extras+=("${bin_dir}")
  elif [[ -n "${LLAMA_CPP_ROOT:-}" && -d "${LLAMA_CPP_ROOT}/build/bin" ]]; then
    extras+=("${LLAMA_CPP_ROOT}/build/bin")
  fi
  # Prefer packaged CUDA matching the vendor build (5080 / cuda_v13 on this host).
  [[ -d /usr/lib/ollama/cuda_v13 ]] && extras+=("/usr/lib/ollama" "/usr/lib/ollama/cuda_v13")
  [[ -d /usr/local/lib/ollama/cuda_v13 ]] && extras+=("/usr/local/lib/ollama" "/usr/local/lib/ollama/cuda_v13")
  [[ -d /usr/local/lib/ollama/cuda_v12 ]] && extras+=("/usr/local/lib/ollama/cuda_v12")
  [[ -d /usr/local/lib/ollama ]] && extras+=("/usr/local/lib/ollama")
  if [[ ${#extras[@]} -gt 0 ]]; then
    local joined
    joined="$(IFS=:; echo "${extras[*]}")"
    export LD_LIBRARY_PATH="${joined}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
}

# L1/L2 CUDA gates — prefer patched vendor llama-server (fork QJL/Polar) when built.
# WHY fallback: Makefile.sync pin may advance (e.g. 8f114a9b) before vendor/<pin>
# is materialised; an older built vendor tree still runs L2 profile A/B.
l1_vendor_llama_cpp_root() {
  local repo="${1:?}"
  local pin vendor cand
  pin="$(grep '^FETCH_HEAD=' "${repo}/Makefile.sync" 2>/dev/null | cut -d= -f2 || echo c84b3020)"
  vendor="${repo}/vendor/llama-cpp-${pin}"
  if [[ -x "${vendor}/build/bin/llama-server" ]]; then
    echo "${vendor}"
    return 0
  fi
  # Newest mtime among built vendor trees (portable: prefer pin match first above).
  cand="$(ls -1dt "${repo}"/vendor/llama-cpp-*/build/bin/llama-server 2>/dev/null | head -1 || true)"
  if [[ -n "${cand}" && -x "${cand}" ]]; then
    echo "$(cd "$(dirname "${cand}")/../.." && pwd)"
    return 0
  fi
  return 1
}

l1_export_llama_binary_env() {
  local repo="${1:?}"
  local root parent
  parent="$(cd "${repo}/.." && pwd)"
  if [[ -n "${LLAMA_CPP_ROOT:-}" && -x "${LLAMA_CPP_ROOT}/build/bin/llama-server" ]]; then
    root="${LLAMA_CPP_ROOT}"
  elif root="$(l1_vendor_llama_cpp_root "${repo}")"; then
    :
  elif [[ -x "${parent}/llama.cpp/build/bin/llama-server" ]]; then
    root="${parent}/llama.cpp"
  else
    echo "L1: no llama-server — run: make -f Makefile.sync apply-patches && ./scripts/build/build_llama_server.sh" >&2
    return 1
  fi
  export LLAMA_CPP_ROOT="${root}"
  export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${root}/build/bin/llama-server}"
  # WHY always match .so to binary: 5080_env may set sibling libllama while bin is vendor.
  export LLAMA_CPP_LIB="${root}/build/bin/libllama.so"
}

llama_server_supports_fork() {
  local bin="${1:-${LLAMA_SERVER_BIN:-}}"
  local help_text
  [[ -x "${bin}" ]] || return 1
  linux_runtime_export_llama_ld_path
  # Match runtime/llama_fork.probe_fork_llama_server: KV types only (stock may
  # advertise --ctx-checkpoints without QJL/Polar/TBQ).
  # WHY no `cmd | grep -q` under pipefail: grep -q closes early → SIGPIPE 141.
  help_text="$("${bin}" --help 2>&1 || true)"
  grep -qE 'qjl1_256|q4_polar|tbq3_0|tbq4_0' <<<"${help_text}"
}

# profile: 0 = OFF baseline (stock q8_0); 1 = ON leg (fork auto when binary supports it).
l1_apply_fork_env() {
  local profile="${1:-1}"
  case "${L1_LLAMA_FORK:-auto}" in
    0|off|false|no|stock)
      export ZEROLLAMA_LLAMA_FORK=0
      ;;
    1|on|fork|eliza)
      export ZEROLLAMA_LLAMA_FORK=1
      ;;
    auto|*)
      if [[ "${profile}" == "0" ]]; then
        export ZEROLLAMA_LLAMA_FORK=0
      else
        unset ZEROLLAMA_LLAMA_FORK
        if ! llama_server_supports_fork "${LLAMA_SERVER_BIN:-}"; then
          export ZEROLLAMA_LLAMA_FORK=0
        fi
      fi
      ;;
  esac
}
