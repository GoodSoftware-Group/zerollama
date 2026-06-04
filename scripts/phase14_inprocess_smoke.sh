#!/usr/bin/env bash
# Phase 14 ctypes GPU sign-off (5080-class default path).
#
# Prerequisite — serve with inprocess on the same GGUF:
#   source ./scripts/phase14_serve_env.sh
#   export LLAMA_MODEL=/path/to/small.q8_0.gguf
#   export LLAMA_CPP_LIB=$HOME/llama.cpp/build/bin/libllama.so
#   export ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess
#   ./zerollama serve
#
# Terminal B:
#   export LLAMA_MODEL=/path/to/same.gguf
#   export RUN_E2E_PROXY_MODEL=<pulled-local-tag>   # optional render-chat
#   ./scripts/phase14_inprocess_smoke.sh
#
# Asserts llama_backend=inprocess and llama_backend_source=env (ROADMAP exit #3).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RUN_E2E_INPROCESS=1
export RUN_E2E_LLAMA_BACKEND_SOURCE=env
exec "${ROOT}/scripts/phase14_backend_smoke.sh"
