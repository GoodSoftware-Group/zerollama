#!/usr/bin/env bash
# Quick bisect: multiseq (profile=1) vs single-seq (profile=0) crash repro.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/runtime_smoke_lib.sh"
source "${ROOT}/scripts/macos_runtime_serve_lib.sh"

export LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-/Users/user1/Sites/inference/llama.cpp}"
export LLAMA_CPP_LIB="${LLAMA_CPP_LIB:-${LLAMA_CPP_ROOT}/build/bin/libllama.dylib}"
export ZEROLLAMA_INFER_TRACE=1
export ZEROLLAMA_KV_NATIVE_DECODE=1
export ZEROLLAMA_KV_NATIVE_SAMPLE=1
export MACOS_RT_HEALTH_MAX=180

smoke_m3_resolve_signoff_model
PROFILE="${1:-0}"
LOOPS="${2:-10}"
export ZEROLLAMA_GPU_PROFILE="${PROFILE}"

lsof -ti :8081 2>/dev/null | xargs kill 2>/dev/null || true
sleep 1
macos_runtime_start_sidecar "$LLAMA_MODEL" "" 1

nseq=$(curl -sf http://127.0.0.1:8081/health | python3 -c "import json,sys; print(json.load(sys.stdin).get('kv_inprocess_n_seq_max'))")
echo "ZEROLLAMA_GPU_PROFILE=${PROFILE} kv_inprocess_n_seq_max=${nseq}"

export RUNTIME_URL=http://127.0.0.1:8081 RUN_E2E_GGUF="$LLAMA_MODEL"
ok=0
for i in $(seq 1 "${LOOPS}"); do
  if bash "${ROOT}/scripts/phase15_metal_crash_repro.sh" runtime_loop 2>/dev/null | tail -1 | grep -q PASS; then
    ok=$((ok+1))
    echo "loop $i ok"
  else
    echo "FAIL at loop $i"
    rg 'infer_trace decode|engine\.(reload|reuse)' /tmp/macos-runtime.log | tail -15
    exit 1
  fi
done
echo "PASS ${LOOPS}/${LOOPS} invocations profile=${PROFILE} n_seq_max=${nseq}"
