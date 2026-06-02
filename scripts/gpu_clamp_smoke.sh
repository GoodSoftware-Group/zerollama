#!/usr/bin/env bash
# Phase 13 VRAM clamp smoke on a GPU host (5080-class).
#
# Serve must be started with clamp on, e.g.:
#   export ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX=auto
#   zerollama serve
#
#   export LLAMA_MODEL LLAMA_SERVER_BIN RUN_E2E_GGUF
#   ./scripts/gpu_clamp_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime_smoke_lib.sh"

export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:8080}"
export ZEROLLAMA_RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"

health_json=$(runtime_fetch_health)
python3 -c "
import json, sys
h = json.loads(sys.argv[1])
policy = h.get('vram_num_ctx_policy') or {}
if not policy.get('clamp_enabled'):
    print(
        'VRAM clamp is off on this daemon. Restart serve with:\\n'
        '  export ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX=auto\\n'
        '  zerollama serve',
        file=sys.stderr,
    )
    sys.exit(1)
print('vram_num_ctx_policy.clamp_enabled: True')
" "$health_json"

if [[ -z "${LLAMA_MODEL:-}" ]] && [[ -z "${RUN_E2E_GGUF:-}" ]]; then
  echo "LLAMA_MODEL or RUN_E2E_GGUF required (GGUF path for runtime generate)" >&2
  exit 1
fi

runtime_resume_if_needed "$health_json"
smoke_prepare_vram_for_runtime
runtime_resume_if_needed

echo "== runtime clamp smoke =="
RUN_E2E_GPU=1 RUN_E2E_PROXY=0 RUN_E2E_VRAM_CLAMP=1 \
  "${ROOT}/scripts/e2e_runtime_smoke.sh"

echo "PASS: gpu_clamp_smoke"
