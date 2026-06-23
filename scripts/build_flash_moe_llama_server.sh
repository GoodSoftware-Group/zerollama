#!/usr/bin/env bash
# Build anemll-flash-llama.cpp llama-server (Metal + Flash-MoE slot-bank).
#
# Output: build/flash-moe-llama-server-darwin/bin/llama-server
#
# Usage:
#   ./scripts/build_flash_moe_llama_server.sh
#   FLASH_MOE_REPO=~/Sites/inference/anemll-flash-llama.cpp ./scripts/build_flash_moe_llama_server.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLASH_MOE_REPO="${FLASH_MOE_REPO:-${HOME}/Sites/inference/anemll-flash-llama.cpp}"
BUILD_DIR="${ROOT}/build/flash-moe-llama-server-darwin"
OUT_BIN="${BUILD_DIR}/bin/llama-server"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "build_flash_moe_llama_server: Darwin only" >&2
  exit 1
fi

if [[ ! -f "${FLASH_MOE_REPO}/CMakeLists.txt" ]]; then
  echo "build_flash_moe_llama_server: missing ${FLASH_MOE_REPO}" >&2
  echo "  git clone --branch Server-Flash-Moe --depth 1 https://github.com/Anemll/anemll-flash-llama.cpp.git ${FLASH_MOE_REPO}" >&2
  exit 1
fi

# shellcheck source=scripts/mac_cgo_env.sh
source "${ROOT}/scripts/mac_cgo_env.sh"
mac_cgo_env
export CC="${CC:-$(xcrun --find clang)}"
export CXX="${CXX:-$(xcrun --find clang++)}"

JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"

echo ">>> Flash-MoE llama-server @ ${FLASH_MOE_REPO}" >&2
cmake -S "${FLASH_MOE_REPO}" -B "${BUILD_DIR}" \
  -DGGML_METAL=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_FLASH_MOE_GPU_BANK=ON \
  -DCMAKE_C_COMPILER="${CC}" \
  -DCMAKE_CXX_COMPILER="${CXX}"

cmake --build "${BUILD_DIR}" --target llama-server -j"${JOBS}"

if [[ ! -x "${OUT_BIN}" ]]; then
  echo ">>> expected ${OUT_BIN}" >&2
  exit 1
fi

echo ">>> OK: ${OUT_BIN}" >&2
"${OUT_BIN}" --version 2>/dev/null || true

echo ">>> serve with Flash-MoE sidecar:" >&2
cat >&2 <<EOF
  export ZEROLLAMA_FLASH_MOE=1
  export ZEROLLAMA_FLASH_MOE_SIDECAR=/path/to/sidecar
  export ZEROLLAMA_LLAMA_SERVER=1
  ${ROOT}/zerollama serve --llama-server-backend
EOF
