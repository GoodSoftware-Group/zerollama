#!/usr/bin/env bash
# L3 full gate — platform dispatcher.
#
# Usage:
#   CUDA_LLAMA_MODEL=/root/eliza-1-9b-256k.gguf ./scripts/l3_full_gate.sh   # Linux CUDA
#   M3_LLAMA_MODEL=/path/to/model.gguf ./scripts/l3_full_gate.sh             # Darwin smoke
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "$(uname -s)" in
  Darwin)
    export M3_LLAMA_MODEL="${M3_LLAMA_MODEL:-${CUDA_LLAMA_MODEL:-${LLAMA_MODEL:-}}}"
    if [[ -z "${M3_LLAMA_MODEL:-}" ]]; then
      echo "Set M3_LLAMA_MODEL (or CUDA_LLAMA_MODEL) for L3 cache smoke" >&2
      exit 1
    fi
    L3_PREFIX_REPEAT="${L3_PREFIX_REPEAT:-150}" \
    L3_COMPARE_NO_CACHE="${L3_COMPARE_NO_CACHE:-1}" \
      "${ROOT}/scripts/l3_cache_smoke.sh"
    "${ROOT}/scripts/l3_gate_report.sh" "${L3_OUT:-/tmp/l3-cache-smoke.json}"
    ;;
  Linux)
    "${ROOT}/scripts/l3_cuda_full_gate.sh"
    ;;
  *)
    echo "unsupported platform for l3_full_gate: $(uname -s)" >&2
    exit 1
    ;;
esac
