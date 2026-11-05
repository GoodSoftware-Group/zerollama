#!/usr/bin/env bash
# Build gliner-server against vendor GLiNER.cpp + ONNX Runtime.
#
# WHY not llama-server: different engine (ORT NER); prod LLAMA_SERVER_BIN stays chat/Laya.
#
# Usage:
#   ./scripts/build/build_gliner_server.sh                 # CPU ORT (GL0)
#   GLINER_ORT=cuda ./scripts/build/build_gliner_server.sh # CUDA ORT (GL4)
#
# Requires: cmake, cargo (tokenizers), C++17.
# CUDA (GLINER_ORT=cuda): GPU ORT tarball + cuDNN on LD_LIBRARY_PATH at runtime.
# Upstream -DGPU_CHECK=ON is optional (GLINER_GPU_CHECK=1); default OFF — Finding 8.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
bash "${ROOT}/scripts/vendor/ensure_gliner_server_deps.sh" >/dev/null
GLINER_ROOT="$(bash "${ROOT}/scripts/vendor/ensure_gliner_cpp.sh" | tail -1)"

ORT_VARIANT=cpu
CMAKE_GPU=OFF
BUILD_DIR="${ROOT}/gliner/server/build"
case "${GLINER_ORT:-cpu}" in
  cpu|CPU) ;;
  cuda|CUDA|gpu|GPU)
    ORT_VARIANT=gpu
    BUILD_DIR="${ROOT}/gliner/server/build-cuda"
    # WHY not GPU_CHECK=ON: upstream find_library(cudnn) fails on this host even when
    # libcudnn.so.9 exists under pip/ollama paths. GPU ORT + --device-id is enough (Finding 8).
    CMAKE_GPU=OFF
    ;;
  *)
    echo "error: GLINER_ORT must be cpu|cuda (got ${GLINER_ORT})" >&2
    exit 1
    ;;
esac

ORT_ROOT="$(ONNXRUNTIME_VARIANT="${ORT_VARIANT}" bash "${ROOT}/scripts/vendor/ensure_onnxruntime.sh" | tail -1)"
JOBS="${ZEROLLAMA_GLINER_JOBS:-$(nproc 2>/dev/null || echo 8)}"

if ! command -v cargo >/dev/null 2>&1; then
  echo "error: cargo required (Rust tokenizers submodule). Install: https://rustup.rs" >&2
  exit 1
fi

# Runtime cuDNN for GPU ORT (pip nvidia-cudnn or ollama mlx_cuda tree).
CUDNN_LIB_DIRS=(
  /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib
  /usr/lib/ollama/mlx_cuda_v12
)
for d in "${CUDNN_LIB_DIRS[@]}"; do
  if [[ -d "${d}" ]]; then
    export LD_LIBRARY_PATH="${d}:${LD_LIBRARY_PATH:-}"
  fi
done

export PATH="/usr/local/cuda/bin:${PATH:-}"
export LD_LIBRARY_PATH="${ORT_ROOT}/lib:/root/nvidia-host:${LD_LIBRARY_PATH:-}"

echo ">>> cmake gliner-server (GLiNER=${GLINER_ROOT} ORT=${ORT_ROOT} variant=${ORT_VARIANT})" >&2
CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DGLINER_CPP_ROOT="${GLINER_ROOT}"
  -DONNXRUNTIME_ROOTDIR="${ORT_ROOT}"
  -DGPU_CHECK=OFF
)
# Optional strict check if operator sets GLINER_GPU_CHECK=1 and cudnn is discoverable.
if [[ "${GLINER_GPU_CHECK:-0}" == "1" ]]; then
  CMAKE_ARGS+=(-DGPU_CHECK=ON)
fi

cmake -S "${ROOT}/gliner/server" -B "${BUILD_DIR}" "${CMAKE_ARGS[@]}"

echo ">>> build gliner-server -j${JOBS}" >&2
cmake --build "${BUILD_DIR}" --target gliner-server -j"${JOBS}"

BIN="${BUILD_DIR}/gliner-server"
if [[ ! -x "${BIN}" ]]; then
  BIN="${BUILD_DIR}/bin/gliner-server"
fi
[[ -x "${BIN}" ]] || { echo "error: expected gliner-server binary" >&2; exit 1; }

echo ">>> OK ${BIN}" >&2
echo ">>> export ZEROLLAMA_GLINER_SERVER_BIN=${BIN}" >&2
if [[ "${ORT_VARIANT}" == "gpu" ]]; then
  echo ">>> CUDA ORT: run with --device-id 0; LD_LIBRARY_PATH must include ORT+cuDNN libs" >&2
fi
echo "${BIN}"
