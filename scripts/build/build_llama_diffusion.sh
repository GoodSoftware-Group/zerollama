#!/usr/bin/env bash
# Build DiffusionGemma CLI + server from the sibling vendor tree (PR #24427).
#
# WHY not build_llama_server.sh / b10615: different binary + draft kernels;
#   prod LLAMA_SERVER_BIN must stay on the chat/Laya/CLM pin.
# WHY CMAKE_CUDA_ARCHITECTURES=120-real: RTX 5080 (Blackwell); wrong arch → no kernel image.
#
# Usage:
#   ./scripts/build/build_llama_diffusion.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIFF_ROOT="$(bash "${ROOT}/scripts/vendor/ensure_diffusion_llama.sh" | tail -1)"
BUILD_DIR="${DIFF_ROOT}/build"
ARCH="${CMAKE_CUDA_ARCHITECTURES:-120-real}"
JOBS="${ZEROLLAMA_DIFFUSION_JOBS:-$(nproc 2>/dev/null || echo 8)}"

export PATH="/usr/local/cuda/bin:${PATH:-}"
export LD_LIBRARY_PATH="/root/nvidia-host:${LD_LIBRARY_PATH:-}"

echo ">>> cmake @ ${DIFF_ROOT} (CUDA ${ARCH})" >&2
cmake -S "${DIFF_ROOT}" -B "${BUILD_DIR}" \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${ARCH}" \
  -DCMAKE_BUILD_TYPE=Release

echo ">>> build llama-diffusion-gemma-cli + llama-diffusion-gemma-server -j${JOBS}" >&2
cmake --build "${BUILD_DIR}" --target llama-diffusion-gemma-cli llama-diffusion-gemma-server -j"${JOBS}"

CLI="${BUILD_DIR}/bin/llama-diffusion-gemma-cli"
SRV="${BUILD_DIR}/bin/llama-diffusion-gemma-server"
if [[ ! -x "${CLI}" || ! -x "${SRV}" ]]; then
  echo "error: expected ${CLI} and ${SRV}" >&2
  exit 1
fi

echo ">>> OK CLI=${CLI}" >&2
echo ">>> OK SRV=${SRV}" >&2
echo ">>> export ZEROLLAMA_DIFFUSION_BIN=${CLI}" >&2
echo ">>> export ZEROLLAMA_DIFFUSION_SERVER_BIN=${SRV}" >&2
echo "${CLI}"
