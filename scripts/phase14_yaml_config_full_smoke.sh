#!/usr/bin/env bash
# Phase 14 optional #6: YAML llama_backend sign-off without editing repo single_gpu.yaml.
#
# Creates a temp config from runtime/configs/single_gpu.yaml (llama_backend: inprocess),
# starts zerollama serve with ZEROLLAMA_RUNTIME_CONFIG only (no backend env), runs
# phase14_yaml_config_smoke.sh, then stops the serve it started.
#
#   export LLAMA_MODEL=/path/to/small.q8_0.gguf
#   export LLAMA_CPP_LIB=$HOME/llama.cpp/build/bin/libllama.so
#   ./scripts/phase14_yaml_config_full_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${LLAMA_MODEL:-}" ]]; then
  echo "Set LLAMA_MODEL to a small GGUF on this host" >&2
  exit 1
fi
if [[ -z "${LLAMA_CPP_LIB:-}" ]]; then
  echo "Set LLAMA_CPP_LIB for inprocess from YAML (ctypes libllama.so)" >&2
  exit 1
fi

pkill -f "${ROOT}/zerollama serve" 2>/dev/null || pkill -f './zerollama serve' 2>/dev/null || true
sleep 2

# shellcheck source=scripts/phase14_serve_env.sh
source "${ROOT}/scripts/phase14_serve_env.sh"
# shellcheck source=scripts/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime_smoke_lib.sh"

RUNTIME_HEALTH_URL="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"
TMPYAML="$(mktemp /tmp/zerollama-phase14-yaml-XXXX.yaml)"
cleanup() {
  pkill -f "${ROOT}/zerollama serve" 2>/dev/null || pkill -f './zerollama serve' 2>/dev/null || true
  rm -f "$TMPYAML"
}
trap cleanup EXIT

sed 's/^# llama_backend: inprocess/llama_backend: inprocess/' \
  "${ROOT}/runtime/configs/single_gpu.yaml" >"$TMPYAML"

echo "== Phase 14 YAML config full smoke (temp ${TMPYAML}) =="

export ZEROLLAMA_RUNTIME_CONFIG="$TMPYAML"
unset ZEROLLAMA_RUNTIME_LLAMA_BACKEND
export LLAMA_CPP_LIB LLAMA_MODEL
: > /tmp/zerollama-phase14-yaml-serve.log
(
  cd "${ROOT}"
  env -u ZEROLLAMA_RUNTIME_URL -u ZEROLLAMA_RUNTIME_LLAMA_BACKEND \
    ZEROLLAMA_RUNTIME_CONFIG="$TMPYAML" \
    ZEROLLAMA_RUNTIME_EMBED="${ZEROLLAMA_RUNTIME_EMBED:-on}" \
    ZEROLLAMA_RUNTIME="${ZEROLLAMA_RUNTIME:-1}" \
    OLLAMA_NO_CLOUD="${OLLAMA_NO_CLOUD:-true}" \
    OLLAMA_TRAINING="${OLLAMA_TRAINING:-false}" \
    LLAMA_MODEL="${LLAMA_MODEL}" \
    LLAMA_CPP_LIB="${LLAMA_CPP_LIB}" \
    ZEROLLAMA_REPO="${ZEROLLAMA_REPO:-$ROOT}" \
    ./zerollama serve >> /tmp/zerollama-phase14-yaml-serve.log 2>&1
) &

got=""
got_src=""
for _ in $(seq 1 60); do
  if curl -sf -m 3 "${RUNTIME_HEALTH_URL}/health" -o /tmp/phase14-yaml-health.json 2>/dev/null; then
    read -r got got_src < <(
      python3 -c "
import json
h = json.load(open('/tmp/phase14-yaml-health.json'))
print((h.get('llama_backend') or ''), (h.get('llama_backend_source') or ''))
"
    )
    if [[ "$got" == "inprocess" && "$got_src" == "config" ]]; then
      echo "serve ready: llama_backend=${got} llama_backend_source=${got_src}"
      break
    fi
  fi
  sleep 2
done

if [[ "${got:-}" != "inprocess" || "${got_src:-}" != "config" ]]; then
  echo "serve failed to reach inprocess+config; log:" >&2
  tail -40 /tmp/zerollama-phase14-yaml-serve.log >&2
  exit 1
fi

env -u ZEROLLAMA_RUNTIME_URL -u RUN_E2E_INPROCESS -u RUN_E2E_LLAMA_CPP_PYTHON \
  LLAMA_MODEL="${LLAMA_MODEL}" \
  "${ROOT}/scripts/phase14_yaml_config_smoke.sh"

echo "PASS: phase14_yaml_config_full_smoke"
