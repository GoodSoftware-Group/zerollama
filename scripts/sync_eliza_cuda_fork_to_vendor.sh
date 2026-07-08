#!/usr/bin/env bash
# Copy eliza CUDA fork KV wiring (SET_ROWS + fused attn + graph route) from the
# llama.cpp sibling into vendor/llama-cpp-<pin> so default vendor builds match
# LLAMA_CPP_ROOT=/path/to/sibling builds.
#
# Usage:
#   ./scripts/sync_eliza_cuda_fork_to_vendor.sh
#   ./scripts/sync_eliza_cuda_fork_to_vendor.sh /path/to/vendor/llama-cpp-c84b3020
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FETCH_HEAD="$(grep '^FETCH_HEAD=' "${ROOT}/Makefile.sync" 2>/dev/null | cut -d= -f2 || echo c84b3020)"
VENDOR="${1:-${ROOT}/vendor/llama-cpp-${FETCH_HEAD}}"
# Always sync FROM the git sibling, never from LLAMA_CPP_ROOT (container mounts vendor as /llama.cpp).
SIBLING="${ELIZA_LLAMA_SIBLING:-${ROOT}/../llama.cpp}"

if [[ ! -f "${SIBLING}/CMakeLists.txt" ]]; then
  echo "error: sibling llama.cpp missing at ${SIBLING}" >&2
  exit 1
fi
if [[ ! -f "${VENDOR}/CMakeLists.txt" ]]; then
  echo "error: vendor missing at ${VENDOR}" >&2
  echo "  run: ./scripts/rebase_vendor_unified.sh --sync" >&2
  exit 1
fi

REL_PATHS=(
  ggml/CMakeLists.txt
  ggml/src/ggml.c
  ggml/src/ggml-cuda/set-rows.cu
  ggml/src/ggml-cuda/qjl.cu
  ggml/src/ggml-cuda/qjl.cuh
  ggml/src/ggml-cuda/ggml-cuda.cu
  ggml/src/ggml-cuda/fused-attn-qjl-tbq.cu
  ggml/src/ggml-cuda/fused-attn-qjl-polar.cu
  ggml/src/ggml-cuda/fused-attn.cu
  ggml/src/ggml-cuda/fused-attn.cuh
  ggml/src/ggml-cuda/polar-set-rows.cuh
  ggml/src/ggml-cuda/qjl-set-rows.cuh
  src/llama-graph.cpp
)

echo ">>> sync eliza CUDA fork: ${SIBLING} → ${VENDOR}"
for rel in "${REL_PATHS[@]}"; do
  src="${SIBLING}/${rel}"
  dst="${VENDOR}/${rel}"
  if [[ ! -f "${src}" ]]; then
    echo "error: missing sibling file ${src}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${dst}")"
  install -m 644 "${src}" "${dst}"
  echo "  ${rel}"
done

if ! grep -q 'GGML_TYPE_QJL1_256' "${VENDOR}/ggml/src/ggml-cuda/set-rows.cu"; then
  echo "error: vendor set-rows.cu missing QJL1_256 after sync" >&2
  exit 1
fi
if ! grep -q 'use_fused_eliza_attn' "${VENDOR}/src/llama-graph.cpp"; then
  echo "error: vendor llama-graph.cpp missing fused route after sync" >&2
  exit 1
fi
if ! grep -q 'qjl_project_q_cuda' "${VENDOR}/ggml/src/ggml-cuda/qjl.cuh"; then
  echo "error: vendor qjl.cuh missing qjl_project_q_cuda after sync" >&2
  exit 1
fi
echo ">>> OK: vendor has CUDA fork KV wiring"
