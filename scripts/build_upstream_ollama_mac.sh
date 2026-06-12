#!/usr/bin/env bash
# Build upstream Ollama llama-server (Metal) + optional go binary for Mac A/B.
#
# Usage:
#   ./scripts/build_upstream_ollama_mac.sh
#   OLLAMA_UPSTREAM_DIR=../ollama-upstream ./scripts/build_upstream_ollama_mac.sh
#   BUILD_UPSTREAM_GO=0 ./scripts/build_upstream_ollama_mac.sh   # llama-server only
set -euo pipefail

ZEROLLAMA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="${OLLAMA_UPSTREAM_DIR:-${ZEROLLAMA_ROOT}/../ollama-upstream}"
BUILD_GO="${BUILD_UPSTREAM_GO:-1}"

if [[ ! -d "${UPSTREAM}/.git" ]]; then
  echo ">>> missing ${UPSTREAM}; run ./scripts/clone_upstream_ollama.sh" >&2
  exit 1
fi

# shellcheck source=scripts/mac_cgo_env.sh
source "${ZEROLLAMA_ROOT}/scripts/mac_cgo_env.sh"
mac_cgo_env
export CC="${CC:-$(xcrun --find clang)}"
export CXX="${CXX:-$(xcrun --find clang++)}"
if [[ "${CXX##*/}" == "clang" ]]; then
  export CXX="$(xcrun --find clang++)"
fi

JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
echo ">>> upstream llama-server (Metal) CC=${CC} CXX=${CXX}" >&2
cmake -S "${UPSTREAM}/llama/server" -B "${UPSTREAM}/build/llama-server-darwin" --preset darwin \
  -DCMAKE_C_COMPILER="${CC}" \
  -DCMAKE_CXX_COMPILER="${CXX}"
cmake --build "${UPSTREAM}/build/llama-server-darwin" --target llama-server llama-quantize -j"${JOBS}"

BIN="${UPSTREAM}/build/llama-server-darwin/bin/llama-server"
if [[ ! -x "${BIN}" ]]; then
  echo ">>> expected ${BIN}" >&2
  exit 1
fi
echo ">>> OK: ${BIN}" >&2
"${BIN}" --version 2>/dev/null || true

if [[ "${BUILD_GO}" == "1" ]]; then
  echo ">>> go build ${UPSTREAM}/ollama" >&2
  (cd "${UPSTREAM}" && go build -o ollama .)
  echo ">>> OK: ${UPSTREAM}/ollama" >&2
fi

echo ">>> serve: OLLAMA_HOST=127.0.0.1:11435 ${UPSTREAM}/ollama serve" >&2
echo ">>> bench: go run ./cmd/bench -host 127.0.0.1:11435 -model MODEL -epochs 3 -format csv" >&2
