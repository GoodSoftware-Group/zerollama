#!/usr/bin/env bash
# Production arm64 macOS build: MLX Metal v3/v4 libs + release zerollama binary.
#
# Output: dist/darwin-arm64/{zerollama,lib/ollama/...}
# Run from dist:  cd dist/darwin-arm64 && ./zerollama serve
#
# Prerequisites:
#   - Xcode + Metal Toolchain:  xcodebuild -downloadComponent MetalToolchain
#   - Local MLX checkouts (override paths below if needed)
#
# Usage:
#   ./scripts/build_production_mac.sh
#   OLLAMA_MLX_SOURCE=../mlx OLLAMA_MLX_C_SOURCE=../mlx-c ./scripts/build_production_mac.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck source=scripts/mac_cgo_env.sh
source "${ROOT}/scripts/mac_cgo_env.sh"
# shellcheck source=scripts/ensure_mlx_sources.sh
source "${ROOT}/scripts/ensure_mlx_sources.sh"

export OLLAMA_MLX_SOURCE="${OLLAMA_MLX_SOURCE:-${ROOT}/../mlx}"
export OLLAMA_MLX_C_SOURCE="${OLLAMA_MLX_C_SOURCE:-${ROOT}/../mlx-c}"

ensure_mlx_sources

# Xcode toolchain first (elan/Homebrew clang breaks C++ MLX linking).
export PATH="/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin:/usr/bin:/bin:${PATH}"

mac_cgo_env_warn_path
mac_cgo_env

export CGO_LDFLAGS="${CGO_LDFLAGS:-} -lc++ -framework Metal -framework Foundation -framework Accelerate -mmacosx-version-min=14.0"

echo ">>> OLLAMA_MLX_SOURCE=${OLLAMA_MLX_SOURCE}" >&2
echo ">>> OLLAMA_MLX_C_SOURCE=${OLLAMA_MLX_C_SOURCE}" >&2
echo ">>> CC=${CC} CXX=${CXX}" >&2

cd "${ROOT}"
./scripts/build_darwin.sh -a arm64 build

OUT="${ROOT}/dist/darwin-arm64/zerollama"
if [[ ! -x "${OUT}" ]]; then
  echo "error: expected ${OUT}" >&2
  exit 1
fi

echo ">>> production build ready" >&2
echo ">>>   cd ${ROOT}/dist/darwin-arm64 && ./zerollama serve" >&2
echo ">>>   ./zerollama doctor" >&2
