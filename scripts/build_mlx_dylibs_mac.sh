#!/usr/bin/env bash
# Build MLX Metal dylibs for macOS arm64 (safetensors / mlxrunner).
#
# WHY this exists: libmlx.dylib + libmlxc.dylib are CMake targets pinned at
# MLX_VERSION / MLX_C_VERSION — not compiled by plain `go build`. Dev installs
# under build/metal-v*/lib/ollama/mlx_metal_v* so repo-root ./zerollama discovers
# them (see x/mlxrunner/mlx/dynamic.go). Production uses INSTALL_PREFIX=dist/darwin-arm64.
#
# Usage:
#   ./scripts/build_mlx_dylibs_mac.sh
#   INSTALL_PREFIX=dist/darwin-arm64 ./scripts/build_mlx_dylibs_mac.sh
#   source ./scripts/build_mlx_dylibs_mac.sh && build_mlx_dylibs_mac
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck source=scripts/mac_cgo_env.sh
source "${ROOT}/scripts/mac_cgo_env.sh"
# shellcheck source=scripts/ensure_mlx_sources.sh
source "${ROOT}/scripts/ensure_mlx_sources.sh"

export OLLAMA_MLX_SOURCE="${OLLAMA_MLX_SOURCE:-${ROOT}/../mlx}"
export OLLAMA_MLX_C_SOURCE="${OLLAMA_MLX_C_SOURCE:-${ROOT}/../mlx-c}"
# WHY: cmake runs go generate during MLX configure; -mod=mod avoids vendor/ mismatch abort.
export GOFLAGS=-mod=mod

# INSTALL_PREFIX: CMAKE_INSTALL_PREFIX for all variants when set (production).
# Empty (default): each variant installs under build/metal-v3|v4 for dev discovery.
INSTALL_PREFIX="${INSTALL_PREFIX:-}"
# BUILD_MLX_V4: auto (skip when SDK < 26), 1 (require v4), 0 (skip v4).
BUILD_MLX_V4="${BUILD_MLX_V4:-auto}"

build_mlx_dylibs_mac() {
  ensure_mlx_sources

  # Xcode toolchain first (elan/Homebrew clang breaks C++ MLX linking).
  export PATH="/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin:/usr/bin:/bin:${PATH}"

  mac_cgo_env_warn_path
  mac_cgo_env

  cd "${ROOT}"

  local build_dir=build/metal-v3
  local v3_prefix="${INSTALL_PREFIX:-${ROOT}/build/metal-v3}"

  echo ">>> Building MLX Metal v3 (macOS 14+)" >&2
  echo ">>>   install: ${v3_prefix}/lib/ollama/mlx_metal_v3/" >&2
  cmake -S . -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLX_ENGINE=ON \
    -DOLLAMA_RUNNER_DIR=mlx_metal_v3 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
    -DCMAKE_INSTALL_PREFIX="${v3_prefix}"
  cmake --build "${build_dir}" --target mlx mlxc --parallel
  cmake --install "${build_dir}" --component MLX

  local sdk_major
  sdk_major="$(xcrun --show-sdk-version 2>/dev/null | cut -d. -f1)"
  if [[ "${BUILD_MLX_V4}" == "0" ]]; then
    echo ">>> Skipping MLX Metal v4 (BUILD_MLX_V4=0)" >&2
    return 0
  fi
  if [[ "${sdk_major:-0}" -lt 26 ]]; then
    if [[ "${BUILD_MLX_V4}" == "1" ]]; then
      echo "error: BUILD_MLX_V4=1 but SDK ${sdk_major:-?} < 26 (need Xcode 26+)" >&2
      exit 1
    fi
    echo ">>> Skipping MLX Metal v4 (SDK ${sdk_major:-?} < 26, need Xcode 26+)" >&2
    return 0
  fi

  local v3_deps="${build_dir}/_deps"
  local build_dir_v4=build/metal-v4
  local v4_prefix="${INSTALL_PREFIX:-${ROOT}/build/metal-v4}"

  echo ">>> Building MLX Metal v4 (macOS 26+, NAX)" >&2
  echo ">>>   install: ${v4_prefix}/lib/ollama/mlx_metal_v4/" >&2
  cmake -S . -B "${build_dir_v4}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLX_ENGINE=ON \
    -DOLLAMA_RUNNER_DIR=mlx_metal_v4 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0 \
    -DCMAKE_INSTALL_PREFIX="${v4_prefix}" \
    -DFETCHCONTENT_SOURCE_DIR_MLX="${v3_deps}/mlx-src" \
    -DFETCHCONTENT_SOURCE_DIR_MLX-C="${v3_deps}/mlx-c-src" \
    -DFETCHCONTENT_SOURCE_DIR_JSON="${v3_deps}/json-src" \
    -DFETCHCONTENT_SOURCE_DIR_FMT="${v3_deps}/fmt-src" \
    -DFETCHCONTENT_SOURCE_DIR_METAL_CPP="${v3_deps}/metal_cpp-src"
  cmake --build "${build_dir_v4}" --target mlx mlxc --parallel
  cmake --install "${build_dir_v4}" --component MLX
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  build_mlx_dylibs_mac
fi
