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
#   BUILD_LLAMA_SERVER=auto  (default) ensure patches + build llama-server when missing/stale
#   BUILD_LLAMA_SERVER=1     force llama-server rebuild
#   BUILD_LLAMA_SERVER=0     skip vendor patch check and llama-server build
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

_llama_vendor_paths() {
  FETCH_HEAD="$(grep '^FETCH_HEAD=' "${ROOT}/Makefile.sync" 2>/dev/null | cut -d= -f2 || echo c84b3020)"
  VENDOR="${ROOT}/vendor/llama-cpp-${FETCH_HEAD}"
  LLAMA_SERVER_BIN="${VENDOR}/build/bin/llama-server"
  LLAMA_LIB="${VENDOR}/build/bin/libllama.dylib"
}

_llama_vendor_patched_ok() {
  [[ -f "${VENDOR}/CMakeLists.txt" ]] \
    && grep -q 'llama_ollama_compat::translate_metadata' "${VENDOR}/src/llama-model-loader.cpp" \
    && grep -q 'llama_ollama_compat::translate_clip_metadata' "${VENDOR}/tools/mtmd/clip.cpp"
}

_llama_server_binary_ok() {
  [[ -x "${LLAMA_SERVER_BIN}" && -f "${LLAMA_LIB}" ]] \
    && grep -qF 'kv/seq-copy' < <(strings "${LLAMA_SERVER_BIN}" 2>/dev/null)
}

_should_build_llama_server() {
  case "${BUILD_LLAMA_SERVER}" in
    0) return 1 ;;
    1) return 0 ;;
    auto|yes)
      if [[ ! -f "${VENDOR}/CMakeLists.txt" ]]; then
        echo ">>> BUILD_LLAMA_SERVER=${BUILD_LLAMA_SERVER}: vendor missing at ${VENDOR}" >&2
        echo ">>>   subprocess models (qwen3.6 MTP): ./scripts/rebase_vendor_unified.sh --apply --sync" >&2
        return 1
      fi
      if ! _llama_vendor_patched_ok; then
        echo ">>> BUILD_LLAMA_SERVER=${BUILD_LLAMA_SERVER}: vendor missing Ollama compat hooks" >&2
        return 0
      fi
      if ! _llama_server_binary_ok; then
        if [[ -x "${LLAMA_SERVER_BIN}" ]]; then
          echo ">>> BUILD_LLAMA_SERVER=${BUILD_LLAMA_SERVER}: llama-server stale or missing /kv/seq-copy" >&2
        else
          echo ">>> BUILD_LLAMA_SERVER=${BUILD_LLAMA_SERVER}: llama-server not built" >&2
        fi
        return 0
      fi
      local stamp="${VENDOR}/build/.zerollama_vendor_rev"
      local head
      head="$(git -C "${VENDOR}" rev-parse HEAD 2>/dev/null || echo unknown)"
      if [[ -f "${stamp}" && "$(<"${stamp}")" == "${head}" ]]; then
        echo ">>> BUILD_LLAMA_SERVER=${BUILD_LLAMA_SERVER}: patched vendor + llama-server OK (${head:0:12})" >&2
        return 1
      fi
      echo ">>> BUILD_LLAMA_SERVER=${BUILD_LLAMA_SERVER}: vendor @ ${head:0:12} newer than last llama-server build" >&2
      return 0
      ;;
    *)
      echo "error: BUILD_LLAMA_SERVER must be 0, 1, or auto (got ${BUILD_LLAMA_SERVER})" >&2
      exit 1
      ;;
  esac
}

_ensure_and_build_llama_server() {
  BUILD_LLAMA_SERVER="${BUILD_LLAMA_SERVER:-auto}"
  _llama_vendor_paths

  if [[ ! -f "${VENDOR}/CMakeLists.txt" ]]; then
    case "${BUILD_LLAMA_SERVER}" in
      0) return 0 ;;
      1)
        echo "error: vendor missing at ${VENDOR}" >&2
        echo "  run: ./scripts/rebase_vendor_unified.sh --apply --sync" >&2
        exit 1
        ;;
      auto|yes)
        echo ">>> warn: vendor missing at ${VENDOR}; skipping llama-server (subprocess MTP unavailable)" >&2
        return 0
        ;;
    esac
  fi

  echo ">>> ensuring vendor Ollama patches + compat hooks" >&2
  "${ROOT}/scripts/ensure_llama_vendor_patches.sh" "${VENDOR}"

  if ! _llama_vendor_patched_ok; then
    echo "error: vendor still missing Ollama compat hooks after ensure_llama_vendor_patches.sh" >&2
    exit 1
  fi

  if _should_build_llama_server; then
    echo ">>> building llama-server (BUILD_LLAMA_SERVER=${BUILD_LLAMA_SERVER})" >&2
    LLAMA_CPP_ROOT="${VENDOR}" "${ROOT}/scripts/build_llama_server.sh"
  fi

  if [[ "${BUILD_LLAMA_SERVER}" != "0" ]]; then
    if ! _llama_server_binary_ok; then
      echo "error: llama-server build finished but binary probe failed" >&2
      exit 1
    fi
    stamp="${VENDOR}/build/.zerollama_vendor_rev"
    if [[ ! -f "${stamp}" ]]; then
      git -C "${VENDOR}" rev-parse HEAD > "${stamp}" 2>/dev/null || true
    fi
    echo ">>> OK: patched vendor + ${LLAMA_SERVER_BIN}" >&2
  fi
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

_ensure_and_build_llama_server

if [[ "$(uname -s)" == Darwin && -f "${ROOT}/scripts/restore_ane_hook_intree.sh" ]]; then
  echo ">>> syncing in-tree ANE hook sources (CGO llama-common)" >&2
  "${ROOT}/scripts/restore_ane_hook_intree.sh"
fi

GOFLAGS=-mod=mod go build -ldflags="-s -w -X=github.com/ollama/ollama/version.Version=${VERSION}" -o "${OUT}" .
echo ">>> wrote ${OUT} (version ${VERSION})" >&2
