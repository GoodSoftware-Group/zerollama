#!/usr/bin/env bash
# Build zerollama on macOS with CGO (Metal ggml + optional MLX dylibs + embedded libpython).
#
# Why go generate runs here: macOS Metal loads shaders from ggml-metal-embed.metal
# (generated from ggml-metal.metal). If ggml adds kernels without regenerating
# the embed, model load can succeed but first decode crashes (missing unary ops
# such as sigmoid for qwen35 SSM). See docs/qwen35-apple-silicon.md.
#
# MLX (safetensors): when ../mlx is present, BUILD_MLX=auto (default) installs
# libmlx/libmlxc under build/metal-v*/lib/ollama/ so repo-root ./zerollama works.
# Set BUILD_MLX=0 for a fast ggml-only rebuild; BUILD_MLX=1 to force MLX rebuild.
#
# Usage:
#   ./scripts/build_zerollama_mac.sh
#   BUILD_MLX=0 ./scripts/build_zerollama_mac.sh          # ggml only (fast)
#   BUILD_MLX=1 ./scripts/build_zerollama_mac.sh          # force MLX dylib rebuild
#   ./scripts/build_zerollama_mac.sh /path/to/output/binary
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-${ROOT}/zerollama}"
if [[ -z "${VERSION:-}" ]]; then
  if git -C "${ROOT}" describe --tags --first-parent --abbrev=7 --long --dirty --always &>/dev/null; then
    VERSION="$(git -C "${ROOT}" describe --tags --first-parent --abbrev=7 --long --dirty --always | sed -e 's/^v//')"
  else
    VERSION="0.19.0"
  fi
fi
BUILD_MLX="${BUILD_MLX:-auto}"

# shellcheck source=scripts/mac_cgo_env.sh
source "${ROOT}/scripts/mac_cgo_env.sh"
mac_cgo_env_warn_path
mac_cgo_env
mac_cgo_env_prefer_training_embed "${ROOT}"

if [[ ! -x "${ROOT}/.venv-training/bin/python" ]]; then
  echo ">>> note: .venv-training missing — training embed falls back to system python3-embed (3.9 on Xcode); run: MAC_SETUP_TRAINING=1 ./scripts/mac_setup.sh" >&2
fi

_mlx_dev_lib() {
  echo "${ROOT}/build/metal-v3/lib/ollama/mlx_metal_v3/libmlxc.dylib"
}

_should_build_mlx() {
  case "${BUILD_MLX}" in
    0) return 1 ;;
    1) return 0 ;;
    auto)
      local mlx_src="${OLLAMA_MLX_SOURCE:-${ROOT}/../mlx}"
      if [[ ! -d "${mlx_src}/.git" ]]; then
        echo ">>> BUILD_MLX=auto: skip MLX (no checkout at ${mlx_src})" >&2
        echo ">>>   safetensors: ./scripts/ensure_mlx_sources.sh --clone && BUILD_MLX=1 $0" >&2
        return 1
      fi
      if [[ -f "$(_mlx_dev_lib)" ]] && [[ "${MLX_FORCE:-0}" != "1" ]]; then
        echo ">>> BUILD_MLX=auto: MLX dylibs present (MLX_FORCE=1 or BUILD_MLX=1 to rebuild)" >&2
        return 1
      fi
      return 0
      ;;
    *)
      echo "error: BUILD_MLX must be 0, 1, or auto (got ${BUILD_MLX})" >&2
      exit 1
      ;;
  esac
}

echo ">>> CC=${CC}" >&2
echo ">>> SDKROOT=${SDKROOT}" >&2
echo ">>> python3-embed: $(pkg-config --modversion python3-embed)" >&2

cd "${ROOT}"
echo ">>> regenerating ggml-metal-embed.metal" >&2
GOFLAGS=-mod=mod go generate ./ml/backend/ggml/ggml/src/ggml-metal/

if _should_build_mlx; then
  echo ">>> building MLX dylibs (BUILD_MLX=${BUILD_MLX})" >&2
  bash "${ROOT}/scripts/build_mlx_dylibs_mac.sh"
fi

GOFLAGS=-mod=mod go build -ldflags="-s -w -X=github.com/ollama/ollama/version.Version=${VERSION}" -o "${OUT}" .
echo ">>> wrote ${OUT} (version ${VERSION})" >&2
