#!/usr/bin/env bash
# L1 full gate — platform dispatcher (Metal tiers on Darwin, CUDA calibrate+concurrent on Linux).
#
# Usage:
#   ./scripts/phase/l1_full_gate.sh
#   CUDA_LLAMA_MODEL=/root/eliza-1-9b-256k.gguf ./scripts/phase/l1_full_gate.sh   # Linux
#   ZEROLLAMA_RUNTIME_URL=http://127.0.0.1:8081 ./scripts/phase/l1_full_gate.sh   # Darwin optional live /health
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

case "$(uname -s)" in
  Darwin)
    "${ROOT}/scripts/phase/l1_metal_gate.sh"
    ;;
  Linux)
    "${ROOT}/scripts/phase/l1_cuda_full_gate.sh"
    ;;
  *)
    echo "unsupported platform for l1_full_gate: $(uname -s)" >&2
    exit 1
    ;;
esac
