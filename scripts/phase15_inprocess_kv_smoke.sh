#!/usr/bin/env bash
# Phase 15 v6: in-process ctypes generate increments kv_decode_steps (needs Phase 14 inprocess serve).
#
# Prerequisite — same as phase14_inprocess_smoke.sh:
#   source ./scripts/phase14_serve_env.sh
#   export LLAMA_MODEL=/path/to/small.q8_0.gguf
#   export LLAMA_CPP_LIB=$HOME/llama.cpp/build/bin/libllama.so
#   export ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess
#   ./zerollama serve
#
# Terminal B:
#   export LLAMA_MODEL=/path/to/same.gguf
#   ./scripts/phase15_inprocess_kv_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RUN_E2E_INPROCESS=1
export RUN_E2E_LLAMA_BACKEND_SOURCE=env
export RUN_E2E_TOOLS=0
exec "${ROOT}/scripts/phase14_backend_smoke.sh"
