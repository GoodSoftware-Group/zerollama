#!/usr/bin/env bash
# Full CT 1564 re-sign-off — one driver for all tiers in docs/5080-runbook.md
#
# Usage (inside CT 1564):
#   cd ~/zerollama
#   source ./scripts/5080_env.sh
#   ./scripts/5080_resignoff.sh                 # all tiers (stop on first FAIL)
#   ./scripts/5080_resignoff.sh --tier 2        # L1 + L3 only
#   ./scripts/5080_resignoff.sh --radix         # append Radix live after tier 2
#   ./scripts/5080_resignoff.sh --build         # rebuild zerollama + sibling llama-server first
#   ./scripts/5080_resignoff.sh --vendor        # rebuild vendor llama-server (Radix seq-copy)
#   ./scripts/5080_resignoff.sh --no-serve      # assume serve already up (tier 1–2)
#
# WHY: switching between gpu-5080-operator-guide, runbook, L3/L1 docs, and ad-hoc env
# exports was error-prone. This script is the executable runbook.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/5080_env.sh
source "${ROOT}/scripts/5080_env.sh"

TIER="${RESIGNOFF_TIER:-all}"
DO_BUILD=0
DO_VENDOR=0
DO_RADIX=0
NO_SERVE=0

usage() {
  sed -n '2,12p' "$0" | tail -n +2
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage 0 ;;
    --tier) TIER="${2:?}"; shift 2 ;;
    --build) DO_BUILD=1; shift ;;
    --vendor) DO_VENDOR=1; shift ;;
    --radix) DO_RADIX=1; shift ;;
    --no-serve) NO_SERVE=1; shift ;;
    *) echo "unknown arg: $1" >&2; usage 1 ;;
  esac
done

run_tier() {
  local want="$1"
  [[ "${TIER}" == "all" || "${TIER}" == "${want}" ]]
}

echo "== 5080 re-sign-off (tier=${TIER}) =="
5080_print_env

if [[ "${DO_BUILD}" -eq 1 ]]; then
  echo "== build zerollama =="
  5080_build_zerollama
  echo "== build sibling llama-server =="
  5080_build_sibling_llama_server
  5080_pull_proxy_model
fi

if [[ "${DO_VENDOR}" -eq 1 ]]; then
  echo "== build vendor llama-server (Radix /kv/seq-copy) =="
  5080_build_vendor_llama_server
fi

if run_tier 0; then
  echo "== tier 0: sanity (no GPU load) =="
  "${ROOT}/scripts/check_gpu_scripts.sh"
  "${ROOT}/scripts/phase15_kv_native_ci.sh"
  "${ROOT}/scripts/phase15_upstream_kv_watch.sh"
  "${ROOT}/scripts/phase17_l2_pin_status.sh"
  if [[ "${RUN_E2E_PREFLIGHT}" == "1" ]]; then
    "${ROOT}/scripts/phase12_golden_ci.sh"
  fi
fi

5080_start_serve_with_profile() {
  local profile="${1:-auto}"
  if [[ "${profile}" == "0" ]]; then
    echo "== start serve (ZEROLLAMA_GPU_PROFILE=0) =="
    5080_stop_serve
    ZEROLLAMA_GPU_PROFILE=0 5080_start_serve
  else
    echo "== start serve (L1 GPU profile auto) =="
    5080_stop_serve
    unset ZEROLLAMA_GPU_PROFILE
    5080_start_serve
  fi
}

if run_tier 1; then
  if [[ "${NO_SERVE}" -eq 0 ]]; then
    # WHY: embedded runtime reads profile at serve start; rtx-5080 qjl1_256 breaks 1B smoke GGUF.
    5080_start_serve_with_profile 0
  else
    5080_wait_health || {
      echo "serve not healthy on :8081 — drop --no-serve or run 5080_start_serve" >&2
      exit 1
    }
  fi
  echo "== tier 1: Phase 11–13 base =="
  5080_cd_repo
  "${ROOT}/scripts/gpu_5080_session.sh"
fi

if run_tier 2; then
  if [[ "${NO_SERVE}" -eq 0 ]]; then
    5080_start_serve_with_profile auto
  else
    5080_wait_health || {
      echo "serve not healthy on :8081 — drop --no-serve or run 5080_start_serve" >&2
      exit 1
    }
  fi
  echo "== tier 2: L1 + L3 production =="
  5080_cd_repo
  RUN_E2E_L1=1 RUN_E2E_L3=1 CUDA_LLAMA_MODEL="${CUDA_LLAMA_MODEL}" \
    "${ROOT}/scripts/gpu_5080_session.sh"
  if [[ "${DO_RADIX}" -eq 1 ]]; then
    echo "== tier 2b: Radix cross-slot live =="
    if [[ "${DO_VENDOR}" -eq 0 ]] && [[ "${LLAMA_SERVER_BIN}" != *"vendor/llama-cpp-"* ]]; then
      echo "warn: Radix usually needs vendor llama-server — consider --vendor" >&2
    fi
    L3_RADIX_LIVE=1 L3_RADIX_OUT="${L3_RADIX_OUT}" CUDA_LLAMA_MODEL="${CUDA_LLAMA_MODEL}" \
      "${ROOT}/scripts/l3_radix_prefix_smoke.sh"
  fi
fi

if run_tier 3; then
  echo "== tier 3: Phase 15 in-process KV =="
  if [[ -z "${LLAMA_CPP_LIB:-}" || ! -f "${LLAMA_CPP_LIB}" ]]; then
    export LLAMA_CPP_LIB="${LLAMA_CPP_BIN}/libllama.so"
  fi
  if ! nm -D "${LLAMA_CPP_LIB}" 2>/dev/null | grep -q llama_memory_kv_; then
    echo "warn: libllama missing kv-ext — run: 5080_build_patched_libllama" >&2
  fi
  5080_stop_serve
  "${ROOT}/scripts/phase15_inprocess_signoff.sh"
fi

if run_tier 4; then
  echo "== tier 4: Phase 16/17 upstream (individual smokes) =="
  export LLAMA_SERVER_BIN P17_MODEL="${P17_MODEL:-${RUN_E2E_PROXY_MODEL}}"
  5080_stop_serve
  LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN}" P17_MODEL="${P17_MODEL}" \
    "${ROOT}/scripts/phase17_llama_server_smoke.sh"
  LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN}" P17_MODEL="${P17_MODEL}" \
    "${ROOT}/scripts/phase17_linux_auto_smoke.sh"
  P17_NUM_PREDICT=32 LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN}" P16_MODEL="${P17_MODEL}" \
    "${ROOT}/scripts/phase16_edge_smoke.sh"
fi

if run_tier 5; then
  echo "== tier 5: optional periodic (skipped by default — run manually) =="
  echo "  L2: ./scripts/l2_cuda_full_gate.sh"
  echo "  Radix: ./scripts/5080_resignoff.sh --tier 2 --radix --vendor"
  echo "  Training: RUN_E2E_TRAINING_OPS=1 ./scripts/gpu_5080_session.sh"
fi

echo ""
echo "PASS: 5080_resignoff (tier=${TIER})"
echo "artifacts:"
echo "  ${GPU_PHASE13_SNAPSHOT_OUT}"
echo "  /tmp/l1-production-gate/gate.json"
echo "  /tmp/l3-cuda-full-gate/gate.json"
[[ "${DO_RADIX}" -eq 1 ]] && echo "  ${L3_RADIX_OUT}"
