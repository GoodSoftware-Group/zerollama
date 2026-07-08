#!/usr/bin/env bash
# Dual RTX 4090 hardware lane — source once per shell, then run gates or serve.
#
#   source ./scripts/dual_4090_env.sh
#   ./scripts/gpu_lane_session.sh
#
# Production (this host): sidecar at /opt/zerollama, Go on :2083.
# Dev: external runtime on :8081 + ZEROLLAMA_RUNTIME_URL (edge build cannot embed).
#
# Doc: docs/cuda-lanes.md
set -euo pipefail

_Z4090_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-.}")/.." && pwd)"
export Z4090_ROOT="${Z4090_ROOT:-$_Z4090_ROOT}"
export ZEROLLAMA_REPO="${ZEROLLAMA_REPO:-${Z4090_ROOT}}"
export CUDA_LANE="${CUDA_LANE:-dual_4090}"

# shellcheck source=scripts/cuda_common_env.sh disable=SC1091
source "${Z4090_ROOT}/scripts/cuda_common_env.sh"

# --- Topology: tensor parallel across both GPUs ---
export ZEROLLAMA_RUNTIME_CONFIG="${ZEROLLAMA_RUNTIME_CONFIG:-${Z4090_ROOT}/runtime/configs/dual_4090.yaml}"
export OLLAMA_SCHED_SPREAD="${OLLAMA_SCHED_SPREAD:-1}"

# --- Serve layout (override for production systemd) ---
export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:2083}"
export ZEROLLAMA_GO_URL="${ZEROLLAMA_GO_URL:-http://127.0.0.1:2083}"
export ZEROLLAMA_RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"
export ZEROLLAMA_RUNTIME_EMBED="${ZEROLLAMA_RUNTIME_EMBED:-0}"
export ZEROLLAMA_EDGE="${ZEROLLAMA_EDGE:-0}"
export ZEROLLAMA_RUNTIME="${ZEROLLAMA_RUNTIME:-1}"
export ZEROLLAMA_LEGACY_RUNNER="${ZEROLLAMA_LEGACY_RUNNER:-0}"
export ZEROLLAMA_LLAMA_SERVER="${ZEROLLAMA_LLAMA_SERVER:-1}"

# --- Installed runtime tree (production) vs repo checkout (dev) ---
if [[ -d /opt/zerollama/runtime/configs/dual_4090.yaml ]]; then
  export Z4090_INSTALL="${Z4090_INSTALL:-/opt/zerollama}"
  export ZEROLLAMA_RUNTIME_CONFIG="${ZEROLLAMA_RUNTIME_CONFIG:-${Z4090_INSTALL}/runtime/configs/dual_4090.yaml}"
  export RT_SITE="${RT_SITE:-${Z4090_INSTALL}/runtime/.venv/lib/python3.11/site-packages}"
  export PYTHONPATH="${RT_SITE}:${Z4090_INSTALL}/runtime${PYTHONPATH:+:${PYTHONPATH}}"
fi

# --- Gate fixtures (override per host) ---
export LLAMA_MODEL="${LLAMA_MODEL:-}"
export CUDA_LLAMA_MODEL="${CUDA_LLAMA_MODEL:-${LLAMA_MODEL:-}}"
export RUN_E2E_GGUF="${RUN_E2E_GGUF:-${LLAMA_MODEL:-}}"
export RUN_E2E_PROXY_MODEL="${RUN_E2E_PROXY_MODEL:-llama3.2:3b}"
export RUN_E2E_PREFLIGHT="${RUN_E2E_PREFLIGHT:-0}"
export GPU_PHASE13_SNAPSHOT_OUT="${GPU_PHASE13_SNAPSHOT_OUT:-/tmp/dual4090-session.json}"

export Z4090_ENV_LOADED=1

dual_4090_assert_topology() {
  local count
  count="$(cuda_visible_gpu_count)"
  if [[ "${count}" -lt 2 ]]; then
    echo "dual_4090_env: need 2+ visible GPUs (got ${count}); set CUDA_LANE= to override detect" >&2
    return 1
  fi
}

dual_4090_health_check() {
  local url="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"
  curl -sf "${url%/}/health" | python3 -c "
import json, sys
h = json.load(sys.stdin)
ac = h.get('autoconfig') or {}
gp = h.get('gpu_profile') or {}
print('status:', h.get('status'))
print('autoconfig.pick:', ac.get('pick'))
print('visible_gpu_count:', ac.get('visible_gpu_count'))
print('gpu_profile:', gp.get('id'), gp.get('n_parallel'))
if ac.get('pick') != 'dual_4090':
    sys.exit(2)
if (ac.get('visible_gpu_count') or 0) < 2:
    sys.exit(3)
"
}
