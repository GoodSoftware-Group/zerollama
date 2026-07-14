#!/usr/bin/env bash
# Phase 15 GPU in-process sign-off (self-contained serve restarts).
#
#   1. phase15_inprocess_kv_smoke.sh       — decode hook + e2e kv_decode_steps
#   2. phase15_inprocess_multiseq_smoke.sh — llama_parallel_slots=2 + generate
#
#   export LLAMA_MODEL=/path/to/small.q8_0.gguf
#   export LLAMA_CPP_LIB=$HOME/llama.cpp/build/bin/libllama.so
#   ./scripts/phase/phase15_inprocess_signoff.sh
#   RUN_P15_AUTO_BATCH_ALL=1 ./scripts/phase/phase15_inprocess_signoff.sh  # v46: + auto-batch GPU gate
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/phase/phase15_runtime_kv_env.sh
source "${ROOT}/scripts/phase/phase15_runtime_kv_env.sh"

if [[ -z "${LLAMA_MODEL:-}" ]]; then
  echo "Set LLAMA_MODEL to a small GGUF on this host" >&2
  exit 1
fi
if [[ -z "${LLAMA_CPP_LIB:-}" ]]; then
  echo "Set LLAMA_CPP_LIB for inprocess backends" >&2
  exit 1
fi

export LLAMA_MODEL LLAMA_CPP_LIB
[[ -n "${RUN_E2E_PROXY_MODEL:-}" ]] && export RUN_E2E_PROXY_MODEL

phase15_runtime_kv_env_apply
phase15_runtime_auto_batch_env_apply
if [[ "${PHASE15_BUILD_KV_EXT:-1}" == "1" ]]; then
  phase15_runtime_kv_ext_build
fi

echo "== Phase 15 in-process GPU sign-off =="

echo ""
echo "== [1/2] KV decode hook (single-seq inprocess) =="
"${ROOT}/scripts/phase/phase15_inprocess_kv_smoke.sh"

echo ""
echo "== [2/2] multi-seq shared context (llama_parallel_slots=2) =="
"${ROOT}/scripts/phase/phase15_inprocess_multiseq_smoke.sh"

if [[ "${RUN_P15_AUTO_BATCH_ALL:-0}" == "1" ]]; then
  echo ""
  echo "== [3/3] auto-batch sign-off (non-stream + stream) =="
  if [[ "${ZEROLLAMA_KV_AUTO_BATCH:-0}" != "1" || "${ZEROLLAMA_KV_AUTO_BATCH_STREAM:-0}" != "1" ]]; then
    echo "WARN: embed serve needs ZEROLLAMA_KV_AUTO_BATCH=1 and ZEROLLAMA_KV_AUTO_BATCH_STREAM=1" >&2
    echo "      export RUN_P15_AUTO_BATCH_ALL=1 before signoff (phase15_runtime_auto_batch_env_apply)" >&2
    exit 1
  fi
  "${ROOT}/scripts/phase/phase15_auto_batch_signoff.sh"
fi

echo ""
echo "PASS: phase15_inprocess_signoff"
