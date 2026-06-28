#!/usr/bin/env bash
# Build zerollama's upstream-shaped llama-server (Metal) from llama/server/.
#
# Uses patched vendor/ (elizaOS + Ollama patches). Prefer ./scripts/build_llama_server.sh
# for the direct vendor CMake path (shared dylibs, simpler link).
#
# Output: build/llama-server-darwin/bin/llama-server
#
# Usage:
#   ./scripts/build_ollama_llama_server_darwin.sh
#   BUILD_ZEROLLAMA_GO=1 ./scripts/build_ollama_llama_server_darwin.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_GO="${BUILD_ZEROLLAMA_GO:-0}"

# shellcheck source=scripts/mac_cgo_env.sh
source "${ROOT}/scripts/mac_cgo_env.sh"
# shellcheck source=scripts/llama_unified_vendor_env.sh
source "${ROOT}/scripts/llama_unified_vendor_env.sh"
mac_cgo_env
llama_unified_vendor_prepare "${ROOT}"

export CC="${CC:-$(xcrun --find clang)}"
export CXX="${CXX:-$(xcrun --find clang++)}"
if [[ "${CXX##*/}" == "clang" ]]; then
  export CXX="$(xcrun --find clang++)"
fi

JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
BUILD_DIR="${ROOT}/build/llama-server-darwin"

if [[ "${LLAMA_SERVER_CLEAN:-1}" == "1" ]]; then
  rm -rf "${BUILD_DIR}"
fi

echo ">>> zerollama llama-server (Metal) CC=${CC} CXX=${CXX}" >&2
echo ">>> OLLAMA_LLAMA_CPP_SOURCE=${OLLAMA_LLAMA_CPP_SOURCE}" >&2
cmake -S "${ROOT}/llama/server" -B "${BUILD_DIR}" --preset darwin \
  -DCMAKE_C_COMPILER="${CC}" \
  -DCMAKE_CXX_COMPILER="${CXX}" \
  -DLLAMA_CURL=OFF
cmake --build "${BUILD_DIR}" --target llama-server llama-quantize -j"${JOBS}"

BIN="${BUILD_DIR}/bin/llama-server"
if [[ ! -x "${BIN}" ]]; then
  echo ">>> expected ${BIN}" >&2
  exit 1
fi
echo ">>> OK: ${BIN}" >&2
"${BIN}" --version 2>/dev/null || true

if [[ "${BUILD_GO}" == "1" ]]; then
  echo ">>> go build zerollama" >&2
  "${ROOT}/scripts/build_zerollama_mac.sh" "${ROOT}/zerollama"
fi

echo ">>> serve: ${ROOT}/zerollama serve --llama-server-backend" >&2
echo ">>> or:    ZEROLLAMA_LLAMA_SERVER=1 ${ROOT}/zerollama serve" >&2
