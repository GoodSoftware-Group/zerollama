#!/usr/bin/env bash
# Lab-only m4-prefill serve: FA + q8_0 KV (stock kv_f16 FA path).
# Never binds :11434 / :8081.
#
# Native Q8 FA (ZEROLLAMA_M4_PREFILL_Q8_KV) regresses vs stock kv_f16 on M4 Max
# even at head=64 — leave unset unless experimenting:
#   ZEROLLAMA_M4_PREFILL_Q8_KV=always ./scripts/phase/m4_prefill_lab_serve.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PORT="${OLLAMA_LAB_PORT:-11435}"
HOST="127.0.0.1:${PORT}"
LAB="${M4_PREFILL_LAB:-${ROOT}/.cache/m4-prefill-lab}"
MODELS="${OLLAMA_MODELS:-${LAB}/models}"
LOG="${LAB}/serve-p1.log"
BIN="${ZEROLLAMA_BIN:-${ROOT}/zerollama}"

if [[ "${PORT}" == "11434" || "${PORT}" == "8081" ]]; then
  echo "refusing production port ${PORT}" >&2
  exit 1
fi

if lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "port ${PORT} already in use; stop the lab listener or pick OLLAMA_LAB_PORT" >&2
  lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >&2 || true
  exit 1
fi

mkdir -p "${LAB}" "${MODELS}"

export OLLAMA_HOST="${HOST}"
export OLLAMA_MODELS="${MODELS}"
export ZEROLLAMA_RUNTIME=0
export ZEROLLAMA_RUNTIME_EMBED=0
export ZEROLLAMA_RUNTIME_DARWIN_SIDECAR=0
export ZEROLLAMA_UMA_SCHED=off
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
# Default OFF: paired A/B showed native Q8 FA −5…−10% vs stock kv_f16 on 0.5B.
if [[ -n "${ZEROLLAMA_M4_PREFILL_Q8_KV:-}" ]]; then
  export ZEROLLAMA_M4_PREFILL_Q8_KV
else
  unset ZEROLLAMA_M4_PREFILL_Q8_KV || true
fi
export ZEROLLAMA_M4_PREFILL_TELEMETRY="${ZEROLLAMA_M4_PREFILL_TELEMETRY:-1}"
unset ZEROLLAMA_M4_PREFILL_SWIGLU || true

echo ">>> lab FA+q8 serve ${HOST} models=${MODELS}" >&2
echo ">>> log ${LOG}" >&2
echo ">>> FA + KV=q8_0; Q8_KV=${ZEROLLAMA_M4_PREFILL_Q8_KV:-off} (native opt-in regresses on M4 Max)" >&2

exec "${BIN}" serve >"${LOG}" 2>&1
