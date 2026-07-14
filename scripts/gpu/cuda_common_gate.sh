#!/usr/bin/env bash
# CUDA-common gate — any NVIDIA lane (5080, dual 4090, single 3090, …).
#
# Proves stack wiring without topology-specific L1/L3 tuning or sm_120 build pins.
#
#   source ./scripts/gpu/cuda_common_env.sh   # or lane env (5080_env / dual_4090_env)
#   export LLAMA_MODEL=/path/to/smoke.gguf LLAMA_SERVER_BIN=...
#   ./scripts/gpu/cuda_common_gate.sh
#
# Optional:
#   RUN_E2E_PREFLIGHT=1     — phase12 golden (needs CGO httplib)
#   RUN_E2E_GPU=1           — runtime generate/chat (needs serve + model)
#   RUN_E2E_PROXY=1         — Go proxy smokes (needs OLLAMA_HOST)
#
# Doc: docs/cuda-lanes.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${CUDA_COMMON_ENV_LOADED:-}" ]]; then
  # shellcheck source=scripts/gpu/cuda_common_env.sh disable=SC1091
  source "${ROOT}/scripts/gpu/cuda_common_env.sh"
fi

echo "== CUDA common gate =="
cuda_print_lane_summary
echo ""

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "warn: nvidia-smi not found — skipping GPU probe legs" >&2
else
  nvidia-smi -L
  echo ""
fi

echo "== tier 0: script + KV CI (no GPU load) =="
"${ROOT}/scripts/check_gpu_scripts.sh"
"${ROOT}/scripts/phase/phase15_kv_native_ci.sh"
"${ROOT}/scripts/phase/phase17_l2_pin_status.sh" || true
echo ""

RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"
if curl -sf "${RUNTIME_URL%/}/health" >/dev/null 2>&1; then
  echo "== runtime /health (informational) =="
  curl -s "${RUNTIME_URL%/}/health" | python3 -c "
import json, sys
h = json.load(sys.stdin)
ac = h.get('autoconfig') or {}
gp = h.get('gpu_profile') or {}
print(json.dumps({
  'status': h.get('status'),
  'autoconfig': ac,
  'gpu_profile_id': gp.get('id'),
  'llama_backend': h.get('llama_backend'),
}, indent=2))
"
  echo ""
else
  echo "skip runtime /health (no listener at ${RUNTIME_URL})" >&2
fi

if [[ "${RUN_E2E_PREFLIGHT:-0}" == "1" ]]; then
  echo "== Phase 12 preflight =="
  "${ROOT}/scripts/phase/phase12_golden_ci.sh"
  echo ""
fi

if [[ "${RUN_E2E_GPU:-0}" == "1" || "${RUN_E2E_PROXY:-0}" == "1" ]]; then
  if [[ -z "${LLAMA_MODEL:-}${RUN_E2E_GGUF:-}" ]]; then
    echo "Set LLAMA_MODEL or RUN_E2E_GGUF for GPU smoke legs" >&2
    exit 1
  fi
  if [[ -z "${LLAMA_SERVER_BIN:-}" ]]; then
    cuda_resolve_llama_server_bin || true
  fi
  if [[ -z "${LLAMA_SERVER_BIN:-}" ]]; then
    echo "Set LLAMA_SERVER_BIN for GPU smoke legs" >&2
    exit 1
  fi
  export RUN_E2E_GPU=1
  [[ "${RUN_E2E_PROXY:-0}" == "1" ]] && export RUN_E2E_PROXY=1
  echo "== gpu_smoke_all (CUDA-common inference path) =="
  "${ROOT}/scripts/gpu/gpu_smoke_all.sh"
fi

echo "== CUDA common gate: PASS =="
