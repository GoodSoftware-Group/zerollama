#!/usr/bin/env bash
# Build llama-server from pinned llama.cpp (see runtime/LLAMA_CPP_PIN.md).
#
# LLAMA_CPP_ROOT defaults to ${zerollama_repo}/../llama.cpp (sibling checkout).
# Why sibling: mac_setup/dev_bootstrap clone here; path must not depend on repo nesting depth.
#
# Why validate nvcc: CMAKE may find headers under cuda-12.8 while CUDACXX points at a
# missing cuda-13/bin/nvcc. RTX 5080: CMAKE_CUDA_ARCHITECTURES=120-real (see docs/testing-smoke.md).
# macOS (M3): GGML_METAL=ON, GGML_CUDA=OFF — produces libllama.dylib + llama-server.
set -euo pipefail

_ZEROLLAMA_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${LLAMA_CPP_ROOT:-${_ZEROLLAMA_ROOT}/../llama.cpp}"
BUILD="${ROOT}/build"

if [[ ! -f "${ROOT}/CMakeLists.txt" ]]; then
  echo "llama.cpp not found at ${ROOT}" >&2
  exit 1
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  ZEROLLAMA_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  # shellcheck source=scripts/mac_cgo_env.sh
  source "${ZEROLLAMA_ROOT}/scripts/mac_cgo_env.sh"
  mac_cgo_env
  export CC="${CC:-$(xcrun --find clang)}"
  export CXX="${CXX:-$(xcrun --find clang++)}"
  if [[ "${CXX##*/}" == "clang" ]]; then
    export CXX="$(xcrun --find clang++)"
  fi
  echo "Building llama-server in ${ROOT} (Metal) CC=${CC} CXX=${CXX}"
  rm -rf "${BUILD}"
  cmake -S "${ROOT}" -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DGGML_METAL=ON \
    -DGGML_CUDA=OFF \
    -DLLAMA_CURL=OFF
  cmake --build "${BUILD}" --target llama llama-server -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
  BIN="${BUILD}/bin/llama-server"
  LIB="${BUILD}/bin/libllama.dylib"
  if [[ -x "${BIN}" && -f "${LIB}" ]]; then
    echo "OK: ${BIN}"
    echo "OK: ${LIB}"
    "${BIN}" --version 2>/dev/null || true
  else
    echo "Build finished but ${BIN} or ${LIB} missing" >&2
    exit 1
  fi
  exit 0
fi

echo "Building llama-server in ${ROOT} (CUDA=${GGML_CUDA:-ON})"
if [[ "${GGML_CUDA:-ON}" == "ON" ]]; then
  cuda_bins=()
  if [[ -n "${CUDA_HOME:-}" ]]; then
    cuda_bins+=("${CUDA_HOME}/bin")
  fi
  cuda_bins+=(
    /usr/local/cuda/bin
    /usr/local/cuda-13/bin
    /usr/local/cuda-13.1/bin
    /usr/local/cuda-12.8/bin
    /usr/local/cuda-12/bin
    /usr/local/cuda-12.3/bin
  )
  CUDACXX=""
  for d in "${cuda_bins[@]}"; do
    [[ -x "${d}/nvcc" ]] || continue
    export PATH="${d}:${PATH}"
    CUDACXX="${d}/nvcc"
    export CUDACXX
    export CUDA_HOME="${d%/bin}"
    break
  done
  if [[ -z "${CUDACXX}" ]]; then
    echo "nvcc not found; install the NVIDIA CUDA toolkit or set CUDA_HOME to a tree that contains bin/nvcc" >&2
    echo "  tried: ${cuda_bins[*]}/nvcc" >&2
    echo "  on this host, check: ls /usr/local/cuda*/bin/nvcc" >&2
    exit 1
  fi
  echo "Using CUDACXX=${CUDACXX} (CUDA_HOME=${CUDA_HOME})"
fi
# Default sm_89 (RTX 4090). RTX 5080 (Blackwell): CMAKE_CUDA_ARCHITECTURES=120-real
# needs a toolkit whose nvcc supports sm_120 (often CUDA 12.8+ or 13.x).
CUDA_ARCH="${CMAKE_CUDA_ARCHITECTURES:-89-real}"

rm -rf "${BUILD}"
# WHY LLAMA_BUILD_WEBUI: eliza fork defaults ON; headless Linux builds fail without WebUI assets.
# WHY GGML_CUDA_GRAPHS: L3 prefix cache clears KV slots; zerollama calls
# llama_context_cuda_graph_invalidate to drop stale captured graphs on CUDA.
cmake -S "${ROOT}" -B "${BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA="${GGML_CUDA:-ON}" \
  -DGGML_CUDA_GRAPHS=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
  -DLLAMA_CURL=ON \
  -DLLAMA_BUILD_WEBUI="${LLAMA_BUILD_WEBUI:-ON}"

cmake --build "${BUILD}" --target llama-server -j"$(nproc)"

BIN="${BUILD}/bin/llama-server"
if [[ -x "${BIN}" ]]; then
  echo "OK: ${BIN}"
  "${BIN}" --version 2>/dev/null || true
else
  echo "Build finished but ${BIN} missing" >&2
  exit 1
fi
