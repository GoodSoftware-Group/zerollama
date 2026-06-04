#!/usr/bin/env bash
# Phase 14 wheel CPU sign-off (ROADMAP exit #4).
#
# Prerequisite — serve with llama-cpp-python on the same GGUF:
#   source ./scripts/phase14_serve_env.sh
#   export LLAMA_MODEL=/path/to/small.q8_0.gguf
#   export ZEROLLAMA_RUNTIME_LLAMA_BACKEND=llama-cpp-python
#   ./zerollama serve
#
# Terminal B:
#   export LLAMA_MODEL=/path/to/same.gguf
#   ./scripts/phase14_wheel_cpu_smoke.sh
#
# Asserts llama_backend=llama-cpp-python, llama_backend_source=env, llama_cpp.gpu_mode=cpu.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RUN_E2E_LLAMA_CPP_PYTHON=1
export RUN_E2E_LLAMA_BACKEND_SOURCE=env
exec "${ROOT}/scripts/phase14_backend_smoke.sh"
