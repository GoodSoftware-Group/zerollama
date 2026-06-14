#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/runtime_smoke_lib.sh"
source "${ROOT}/scripts/macos_runtime_serve_lib.sh"

export LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-/Users/user1/Sites/inference/llama.cpp}"
export LLAMA_CPP_LIB="${LLAMA_CPP_LIB:-${LLAMA_CPP_ROOT}/build/bin/libllama.dylib}"
export ZEROLLAMA_INFER_TRACE=1
export ZEROLLAMA_KV_NATIVE_DECODE=1
export ZEROLLAMA_KV_NATIVE_SAMPLE=1
export ZEROLLAMA_GPU_PROFILE=1
export MACOS_RT_HEALTH_MAX=180

smoke_m3_resolve_signoff_model
lsof -ti :8081 2>/dev/null | xargs kill 2>/dev/null || true
sleep 1
macos_runtime_start_sidecar "$LLAMA_MODEL" "" 1
macos_runtime_start_go

export RUN_E2E_GGUF="$LLAMA_MODEL" RUN_E2E_INPROCESS=1
echo "== broker_gguf =="
bash "${ROOT}/scripts/phase15_metal_crash_repro.sh" broker_gguf
echo "== phase14_full (gguf kept) =="
bash "${ROOT}/scripts/phase15_metal_crash_repro.sh" phase14_full
