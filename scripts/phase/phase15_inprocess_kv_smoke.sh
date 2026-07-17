#!/usr/bin/env bash
# Phase 15 v6: in-process ctypes generate increments kv_decode_steps.
#
# Self-contained: starts runtime (embed when non-edge, uv sidecar when edge) with
# inprocess backend, runs Phase 14 backend smoke + /health kv_decode_steps.
#
#   export LLAMA_MODEL=/path/to/small.q8_0.gguf
#   export LLAMA_CPP_LIB=$HOME/llama.cpp/build/bin/libllama.so
#   ./scripts/phase/phase15_inprocess_kv_smoke.sh
#
# Optional: RUN_E2E_PROXY_MODEL=pulled-tag for render-chat tokenize check.
# PHASE15_USE_SIDECAR=1 — force Linux uv sidecar (default auto when edge binary).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -z "${LLAMA_MODEL:-}" ]]; then
  echo "Set LLAMA_MODEL to a small GGUF on this host" >&2
  exit 1
fi
if [[ -z "${LLAMA_CPP_LIB:-}" ]]; then
  echo "Set LLAMA_CPP_LIB for inprocess (ctypes libllama.so)" >&2
  exit 1
fi

# shellcheck source=scripts/phase14_serve_env.sh
source "${ROOT}/scripts/phase14_serve_env.sh"
# shellcheck source=scripts/phase/phase15_runtime_kv_env.sh
source "${ROOT}/scripts/phase/phase15_runtime_kv_env.sh"
# shellcheck source=scripts/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime_smoke_lib.sh"
# shellcheck source=scripts/linux_runtime_serve_lib.sh
source "${ROOT}/scripts/linux_runtime_serve_lib.sh"

phase15_runtime_kv_env_apply

ZEROLLAMA_BIN="${ZEROLLAMA_BIN:-${ROOT}/zerollama}"
# WHY alternate ports: dual-4090 prod often owns :2083 (ollama) + :8081 (runtime).
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:18083}"
ZEROLLAMA_RUNTIME_EMBED_PORT="${ZEROLLAMA_RUNTIME_EMBED_PORT:-18081}"
export OLLAMA_HOST ZEROLLAMA_RUNTIME_EMBED_PORT ZEROLLAMA_BIN
RUNTIME_URL="http://127.0.0.1:${ZEROLLAMA_RUNTIME_EMBED_PORT}"

_use_sidecar=0
_ver_out="$("${ZEROLLAMA_BIN}" -v 2>&1 || true)"
if [[ "${PHASE15_USE_SIDECAR:-}" == "1" ]]; then
  _use_sidecar=1
elif [[ "${_ver_out}" == *"edge build: true"* ]]; then
  # WHY: EdgeBuildTag forces RuntimeEmbedEnabled=false — cannot in-process CPython.
  _use_sidecar=1
fi

_GO_PID=""
cleanup() {
  [[ -n "${_GO_PID}" ]] && kill "${_GO_PID}" 2>/dev/null || true
  if [[ "${_use_sidecar}" -eq 1 ]]; then
    linux_runtime_sidecar_cleanup
    linux_runtime_stop_sidecar_port
  else
    pkill -f "${ZEROLLAMA_BIN} serve" 2>/dev/null || pkill -f "${ROOT}/zerollama serve" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "== Phase 15 in-process KV decode hook smoke =="
echo "OLLAMA_HOST=${OLLAMA_HOST} runtime=${RUNTIME_URL} mode=$([ "${_use_sidecar}" -eq 1 ] && echo sidecar || echo embed)"

if [[ "${_use_sidecar}" -eq 1 ]]; then
  export ZEROLLAMA_RUNTIME_URL="${RUNTIME_URL}"
  export ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess
  export ZEROLLAMA_RUNTIME_EMBED=0
  export LINUX_RT_LOG="${LINUX_RT_LOG:-/tmp/zerollama-phase15-kv-runtime.log}"
  linux_runtime_urls
  linux_runtime_stop_sidecar_port
  export ZEROLLAMA_RUNTIME_CONFIG="${ZEROLLAMA_RUNTIME_CONFIG:-${ROOT}/runtime/configs/single_gpu.yaml}"
  unset ZEROLLAMA_AUTO_CONFIG
  linux_runtime_start_sidecar "${LLAMA_MODEL}" "${ZEROLLAMA_RUNTIME_CONFIG}"

  : > /tmp/zerollama-phase15-kv-serve.log
  (
    cd "${ROOT}"
    env \
      ZEROLLAMA_RUNTIME_URL="${RUNTIME_URL}" \
      ZEROLLAMA_RUNTIME_EMBED=0 \
      ZEROLLAMA_RUNTIME="${ZEROLLAMA_RUNTIME:-1}" \
      OLLAMA_HOST="${OLLAMA_HOST}" \
      OLLAMA_NO_CLOUD="${OLLAMA_NO_CLOUD:-true}" \
      OLLAMA_TRAINING="${OLLAMA_TRAINING:-false}" \
      ZEROLLAMA_EDGE="${ZEROLLAMA_EDGE:-0}" \
      LLAMA_MODEL="${LLAMA_MODEL}" \
      LLAMA_CPP_LIB="${LLAMA_CPP_LIB}" \
      ZEROLLAMA_REPO="${ZEROLLAMA_REPO:-$ROOT}" \
      "${ZEROLLAMA_BIN}" serve >> /tmp/zerollama-phase15-kv-serve.log 2>&1
  ) &
  _GO_PID=$!
  for _ in $(seq 1 30); do
    if curl -sf -m 3 "${OLLAMA_HOST%/}/api/tags" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
else
  pkill -f "${ROOT}/zerollama serve" 2>/dev/null || pkill -f './zerollama serve' 2>/dev/null || true
  sleep 2
  : > /tmp/zerollama-phase15-kv-serve.log
  (
    cd "${ROOT}"
    env -u ZEROLLAMA_RUNTIME_URL \
      ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess \
      ZEROLLAMA_RUNTIME_EMBED="${ZEROLLAMA_RUNTIME_EMBED:-on}" \
      ZEROLLAMA_RUNTIME_EMBED_PORT="${ZEROLLAMA_RUNTIME_EMBED_PORT}" \
      ZEROLLAMA_RUNTIME="${ZEROLLAMA_RUNTIME:-1}" \
      OLLAMA_HOST="${OLLAMA_HOST}" \
      OLLAMA_NO_CLOUD="${OLLAMA_NO_CLOUD:-true}" \
      OLLAMA_TRAINING="${OLLAMA_TRAINING:-false}" \
      ZEROLLAMA_EDGE="${ZEROLLAMA_EDGE:-0}" \
      LLAMA_MODEL="${LLAMA_MODEL}" \
      LLAMA_CPP_LIB="${LLAMA_CPP_LIB}" \
      ZEROLLAMA_REPO="${ZEROLLAMA_REPO:-$ROOT}" \
      "${ZEROLLAMA_BIN}" serve >> /tmp/zerollama-phase15-kv-serve.log 2>&1
  ) &
  _GO_PID=$!
fi

backend=""
backend_src=""
for _ in $(seq 1 60); do
  if curl -sf -m 3 "${RUNTIME_URL}/health" -o /tmp/phase15-kv-health.json 2>/dev/null; then
    read -r backend backend_src < <(
      python3 -c "
import json
h = json.load(open('/tmp/phase15-kv-health.json'))
print((h.get('llama_backend') or ''), (h.get('llama_backend_source') or ''))
"
    )
    if [[ "$backend" == "inprocess" && ( "$backend_src" == "env" || "$backend_src" == "config" ) ]]; then
      echo "serve ready: llama_backend=${backend} llama_backend_source=${backend_src}"
      break
    fi
  fi
  sleep 2
done

if [[ "${backend:-}" != "inprocess" ]]; then
  echo "serve failed to reach inprocess; log:" >&2
  tail -40 /tmp/zerollama-phase15-kv-serve.log >&2
  [[ -f "${LINUX_RT_LOG:-/tmp/zerollama-phase15-kv-runtime.log}" ]] && tail -40 "${LINUX_RT_LOG:-/tmp/zerollama-phase15-kv-runtime.log}" >&2
  exit 1
fi

(
  export RUN_E2E_INPROCESS=1
  export RUN_E2E_LLAMA_BACKEND_SOURCE="${backend_src}"
  export OLLAMA_HOST="${OLLAMA_HOST}"
  export ZEROLLAMA_RUNTIME_URL="${RUNTIME_URL}"
  env -u RUN_E2E_LLAMA_CPP_PYTHON \
    LLAMA_MODEL="${LLAMA_MODEL}" \
    "${ROOT}/scripts/phase/phase14_backend_smoke.sh"
)

smoke_runtime_assert_kv_snapshot "$RUNTIME_URL"

echo "PASS: phase15_inprocess_kv_smoke"
