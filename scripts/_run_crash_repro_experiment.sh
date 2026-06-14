#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime_smoke_lib.sh"
# shellcheck source=scripts/macos_runtime_serve_lib.sh
source "${ROOT}/scripts/macos_runtime_serve_lib.sh"

export LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-/Users/user1/Sites/inference/llama.cpp}"
export LLAMA_CPP_LIB="${LLAMA_CPP_LIB:-${LLAMA_CPP_ROOT}/build/bin/libllama.dylib}"
export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${LLAMA_CPP_ROOT}/build/bin/llama-server}"
export ZEROLLAMA_INFER_TRACE=1
export ZEROLLAMA_KV_NATIVE_DECODE=1
export ZEROLLAMA_KV_NATIVE_SAMPLE=1
export MACOS_RT_HEALTH_MAX=180
export MACOS_RT_LOG="${MACOS_RT_LOG:-/tmp/macos-runtime.log}"

smoke_m3_resolve_signoff_model
echo "MODEL=${LLAMA_MODEL}"

lsof -ti :8081 2>/dev/null | xargs kill 2>/dev/null || true
sleep 1

macos_runtime_start_sidecar "$LLAMA_MODEL" "" 1
macos_runtime_start_go

echo "== health =="
curl -sf "${ZEROLLAMA_RUNTIME_URL%/}/health" | python3 -c "
import json,sys
h=json.load(sys.stdin)
print('llama_backend', h.get('llama_backend'))
print('llama_model', h.get('llama_model'))
dl=(h.get('kv_decode_loop') or {})
print('decode_loop', dl)
"

echo "== running crash repro (all scenarios) =="
"${ROOT}/scripts/phase15_metal_crash_repro.sh" all

echo "== infer_trace tail =="
rg 'infer_trace|engine\.(reload|reuse|start)|create_tensor: loading' "${MACOS_RT_LOG}" | tail -80
