#!/usr/bin/env bash
# Phase 14 optional wheel GPU smoke (llama-cpp-python + ZEROLLAMA_LLAMA_CPP_N_GPU_LAYERS).
#
# WHY separate from inprocess sign-off: pip wheels can abort on GPU decode on some hosts
# while ctypes libllama.so works. Only run after CPU wheel smoke passes (phase14_both_backends).
#
# Terminal A — serve:
#   source ./scripts/phase14_serve_env.sh
#   export LLAMA_MODEL=/path/to/small.q8_0.gguf
#   export ZEROLLAMA_RUNTIME_LLAMA_BACKEND=llama-cpp-python
#   export ZEROLLAMA_LLAMA_CPP_N_GPU_LAYERS=99   # or layer count for your GGUF
#   ./zerollama serve
#
# Terminal B:
#   export LLAMA_MODEL=/path/to/same.gguf
#   ./scripts/phase14_wheel_gpu_smoke.sh
#
# Asserts llama_cpp.gpu_mode=gpu after generate (see RUN_E2E_LLAMA_CPP_PYTHON_GPU).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RUN_E2E_LLAMA_CPP_PYTHON=1
export RUN_E2E_LLAMA_CPP_PYTHON_GPU=1
export RUN_E2E_LLAMA_BACKEND_SOURCE=env
exec "${ROOT}/scripts/phase14_backend_smoke.sh"
