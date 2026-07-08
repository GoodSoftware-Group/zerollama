#!/usr/bin/env bash
# Hardware lane session — CUDA-common gate + lane-specific sign-off tiers.
#
# Auto-detects lane from GPU count / VRAM unless CUDA_LANE is set:
#   dual_4090  — 2+ GPUs, tensor parallel YAML
#   rtx_5080   — 1 GPU ≤18 GiB (16 GB class)
#   single_4090 — 1 GPU ≥20 GiB
#   single_cuda — fallback single-GPU
#
# Usage:
#   ./scripts/gpu_lane_session.sh
#   CUDA_LANE=dual_4090 ./scripts/gpu_lane_session.sh
#   RUN_E2E_L1=1 RUN_E2E_L3=1 ./scripts/gpu_lane_session.sh   # 5080 lane only today
#
# Doc: docs/cuda-lanes.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/cuda_common_env.sh disable=SC1091
source "${ROOT}/scripts/cuda_common_env.sh"

LANE="${CUDA_LANE:-$(cuda_lane_detect)}"
export CUDA_LANE="${LANE}"

echo "== gpu lane session: ${LANE} =="
cuda_print_lane_summary
echo ""

case "${LANE}" in
  dual_4090)
    # shellcheck source=scripts/dual_4090_env.sh disable=SC1091
    source "${ROOT}/scripts/dual_4090_env.sh"
    dual_4090_assert_topology
    echo "== CUDA common gate =="
    RUN_E2E_GPU="${RUN_E2E_GPU:-1}" RUN_E2E_PROXY="${RUN_E2E_PROXY:-1}" \
      "${ROOT}/scripts/cuda_common_gate.sh"
    echo "== dual 4090 topology check =="
    dual_4090_health_check
    echo ""
    echo "Lane-specific gates for dual_4090 (L1 concurrent @ np=8, L3 @ 27k) — run manually:"
    echo "  CUDA_LANE=dual_4090 source scripts/dual_4090_env.sh"
    echo "  CUDA_LLAMA_MODEL=/path/to/9b.gguf ./scripts/l1_cuda_full_gate.sh"
    echo "  CUDA_LLAMA_MODEL=/path/to/9b.gguf ./scripts/l3_cuda_full_gate.sh"
    ;;
  rtx_5080|single_4090|single_cuda)
    if [[ -f "${ROOT}/scripts/5080_env.sh" ]]; then
      export Z5080_AUTO_ENV=1
      "${ROOT}/scripts/gpu_5080_session.sh"
    else
      echo "5080_env.sh missing — running cuda_common_gate only" >&2
      RUN_E2E_GPU="${RUN_E2E_GPU:-1}" "${ROOT}/scripts/cuda_common_gate.sh"
    fi
    ;;
  *)
    echo "unknown CUDA_LANE=${LANE}" >&2
    exit 1
    ;;
esac

echo "== gpu lane session: done (${LANE}) =="
