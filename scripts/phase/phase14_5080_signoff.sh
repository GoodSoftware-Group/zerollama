#!/usr/bin/env bash
# 5080-class Phase 14 + 15 operator sign-off (self-contained serve restarts).
#
# Runs, in order:
#   1. phase14_both_backends.sh       — inprocess (ctypes GPU) + wheel CPU
#   2. phase14_yaml_config_full_smoke — YAML llama_backend (temp config, no repo edit)
#   3. phase15_inprocess_signoff.sh   — KV decode hook + multi-seq + kv-snapshot
#
# Skips wheel GPU (known abort on cu124 wheels); use inprocess for production GPU.
#
#   export LLAMA_MODEL=/path/to/small.q8_0.gguf
#   export LLAMA_CPP_LIB=$HOME/llama.cpp/build/bin/libllama.so
#   ./scripts/phase/phase14_5080_signoff.sh
#
# Optional:
#   RUN_E2E_SKIP_LLAMA_CPP_PYTHON=1  — ctypes-only hosts without pip wheel
#   RUN_E2E_PROXY_MODEL=pulled-tag   — render-chat tokenize in backend smokes
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

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

echo "== Phase 14/15 5080 sign-off suite =="

echo ""
echo "== [1/3] both backends (inprocess + wheel CPU) =="
"${ROOT}/scripts/phase/phase14_both_backends.sh"

echo ""
echo "== [2/3] YAML config (temp single_gpu, llama_backend_source=config) =="
"${ROOT}/scripts/phase/phase14_yaml_config_full_smoke.sh"

echo ""
echo "== [3/3] Phase 15 in-process GPU sign-off =="
"${ROOT}/scripts/phase/phase15_inprocess_signoff.sh"

echo ""
echo "PASS: phase14_5080_signoff"
