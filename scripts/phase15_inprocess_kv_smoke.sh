#!/usr/bin/env bash
# Phase 15 v6: in-process ctypes generate increments kv_decode_steps.
#
# Self-contained: starts embedded serve with inprocess backend, runs Phase 14
# backend smoke (generate + /health kv_decode_steps assertions).
#
#   export LLAMA_MODEL=/path/to/small.q8_0.gguf
#   export LLAMA_CPP_LIB=$HOME/llama.cpp/build/bin/libllama.so
#   ./scripts/phase15_inprocess_kv_smoke.sh
#
# Optional: RUN_E2E_PROXY_MODEL=pulled-tag for render-chat tokenize check.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${LLAMA_MODEL:-}" ]]; then
  echo "Set LLAMA_MODEL to a small GGUF on this host" >&2
  exit 1
fi
if [[ -z "${LLAMA_CPP_LIB:-}" ]]; then
  echo "Set LLAMA_CPP_LIB for inprocess (ctypes libllama.so)" >&2
  exit 1
fi

pkill -f "${ROOT}/zerollama serve" 2>/dev/null || pkill -f './zerollama serve' 2>/dev/null || true
sleep 2

# shellcheck source=scripts/phase14_serve_env.sh
source "${ROOT}/scripts/phase14_serve_env.sh"
# shellcheck source=scripts/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime_smoke_lib.sh"

RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"

cleanup() {
  pkill -f "${ROOT}/zerollama serve" 2>/dev/null || pkill -f './zerollama serve' 2>/dev/null || true
}
trap cleanup EXIT

echo "== Phase 15 in-process KV decode hook smoke =="

: > /tmp/zerollama-phase15-kv-serve.log
(
  cd "${ROOT}"
  env -u ZEROLLAMA_RUNTIME_URL \
    ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess \
    ZEROLLAMA_RUNTIME_EMBED="${ZEROLLAMA_RUNTIME_EMBED:-on}" \
    ZEROLLAMA_RUNTIME="${ZEROLLAMA_RUNTIME:-1}" \
    OLLAMA_NO_CLOUD="${OLLAMA_NO_CLOUD:-true}" \
    OLLAMA_TRAINING="${OLLAMA_TRAINING:-false}" \
    LLAMA_MODEL="${LLAMA_MODEL}" \
    LLAMA_CPP_LIB="${LLAMA_CPP_LIB}" \
    ZEROLLAMA_REPO="${ZEROLLAMA_REPO:-$ROOT}" \
    ./zerollama serve >> /tmp/zerollama-phase15-kv-serve.log 2>&1
) &

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
    if [[ "$backend" == "inprocess" && "$backend_src" == "env" ]]; then
      echo "serve ready: llama_backend=${backend} llama_backend_source=${backend_src}"
      break
    fi
  fi
  sleep 2
done

if [[ "${backend:-}" != "inprocess" || "${backend_src:-}" != "env" ]]; then
  echo "serve failed to reach inprocess+env; log:" >&2
  tail -40 /tmp/zerollama-phase15-kv-serve.log >&2
  exit 1
fi

env -u ZEROLLAMA_RUNTIME_URL -u RUN_E2E_LLAMA_CPP_PYTHON \
  LLAMA_MODEL="${LLAMA_MODEL}" \
  "${ROOT}/scripts/phase14_inprocess_smoke.sh"

smoke_runtime_assert_kv_snapshot "$RUNTIME_URL"

echo "PASS: phase15_inprocess_kv_smoke"
