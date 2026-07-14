#!/usr/bin/env bash
# Phase 14 GPU sign-off for in-process backends (restart serve between backends).
#
# Why restart per backend: ZEROLLAMA_RUNTIME_LLAMA_BACKEND is fixed when the embedded
# Python runtime starts inside zerollama serve — it cannot flip in-process → wheel
# without a new process.
#
# Why env -u ZEROLLAMA_RUNTIME_URL on serve start: phase14_serve_env unsets URL so Go
# embeds :8081; re-exporting URL in this shell broke embed (external sidecar mode).
#
# Why env -u RUN_E2E_INPROCESS between runs: operator shells often export smoke flags;
# wheel run must not inherit RUN_E2E_INPROCESS=1 from a prior inprocess session.
#
#   export LLAMA_MODEL=/path/to/small.q8_0.gguf
#   export LLAMA_CPP_LIB=$HOME/llama.cpp/build/bin/libllama.so   # inprocess only
#   export RUN_E2E_PROXY_MODEL=my-local-tag
#   ./scripts/phase/phase14_both_backends.sh
#
# Skips a backend when RUN_E2E_SKIP_INPROCESS=1 or RUN_E2E_SKIP_LLAMA_CPP_PYTHON=1.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/phase/phase14_serve_env.sh
source "${ROOT}/scripts/phase/phase14_serve_env.sh"

if [[ -z "${LLAMA_MODEL:-}" ]]; then
  echo "Set LLAMA_MODEL to a small GGUF on this host" >&2
  exit 1
fi

export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:8080}"
export OLLAMA_TRAINING="${OLLAMA_TRAINING:-false}"
# Do not export ZEROLLAMA_RUNTIME_URL here — phase14_serve_env unsets it so Go embeds :8081.
RUNTIME_HEALTH_URL="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"

_stop_serve() {
  pkill -f "${ROOT}/zerollama serve" 2>/dev/null || pkill -f './zerollama serve' 2>/dev/null || true
  sleep 2
}

_start_serve() {
  local backend="$1"
  _stop_serve
  export ZEROLLAMA_RUNTIME_LLAMA_BACKEND="$backend"
  : > /tmp/zerollama-phase14-serve.log
  (cd "${ROOT}" && env -u ZEROLLAMA_RUNTIME_URL ./zerollama serve >> /tmp/zerollama-phase14-serve.log 2>&1) &
  for _ in $(seq 1 45); do
    if curl -sf -m 3 "${RUNTIME_HEALTH_URL}/health" -o /tmp/phase14-health.json 2>/dev/null; then
      read -r got got_src < <(
        python3 -c "
import json
h = json.load(open('/tmp/phase14-health.json'))
print((h.get('llama_backend') or ''), (h.get('llama_backend_source') or ''))
"
      )
      if [[ "$got" == "$backend" && "$got_src" == "env" ]]; then
        return 0
      fi
    fi
    sleep 2
  done
  echo "serve failed to start (backend=${backend}); log:" >&2
  tail -30 /tmp/zerollama-phase14-serve.log >&2
  return 1
}

_run_smoke() {
  local script="$1"
  # Unset URL and backend flags (operator shell may export stale RUN_E2E_*).
  # shellcheck disable=SC2086
  env -u ZEROLLAMA_RUNTIME_URL -u RUN_E2E_INPROCESS -u RUN_E2E_LLAMA_CPP_PYTHON \
    LLAMA_MODEL="${LLAMA_MODEL}" \
    "${ROOT}/scripts/${script}"
}

_ran=0

if [[ "${RUN_E2E_SKIP_INPROCESS:-0}" != "1" ]]; then
  if [[ -z "${LLAMA_CPP_LIB:-}" ]]; then
    echo "error: LLAMA_CPP_LIB required for inprocess backend" >&2
    exit 1
  fi
  echo "== Phase 14 inprocess =="
  _start_serve inprocess
  _run_smoke phase14_inprocess_smoke.sh
  _ran=$((_ran + 1))
fi

if [[ "${RUN_E2E_SKIP_LLAMA_CPP_PYTHON:-0}" != "1" ]]; then
  if ! python3 -c "import llama_cpp" 2>/dev/null; then
    echo "skip llama-cpp-python: pip install llama-cpp-python" >&2
  else
    echo "== Phase 14 llama-cpp-python =="
    _start_serve llama-cpp-python
    _run_smoke phase14_wheel_cpu_smoke.sh
    _ran=$((_ran + 1))
  fi
fi

if [[ "$_ran" -lt 1 ]]; then
  echo "error: no Phase 14 backend ran (both skipped or llama_cpp missing)" >&2
  echo "  unset RUN_E2E_SKIP_* or pip install llama-cpp-python" >&2
  exit 1
fi

echo "PASS: phase14_both_backends (${_ran} backend(s))"
