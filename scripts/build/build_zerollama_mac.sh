#!/usr/bin/env bash
# Build zerollama on macOS with CGO (Metal ggml + optional MLX dylibs + embedded libpython).
#
# Why Metal embed regen runs here: ggml-metal-device.m JIT-compiles UTF-8
# kernels/<kind>.metal via _ggml_metallib_*_{start,end}. Stale embeds miss
# new kernels (TQ2/E8/unary) and first decode can crash.
# See docs/qwen35-apple-silicon.md.
#
# MLX (safetensors): when ../mlx is present, BUILD_MLX=auto (default) installs
# libmlx/libmlxc under build/metal-v*/lib/ollama/ so repo-root ./zerollama works.
# Set BUILD_MLX=0 for a fast ggml-only rebuild; BUILD_MLX=1 to force MLX rebuild.
#
# UMA broker client (mlxrunner GPU admission): BUILD_UMA=auto (default) links
# -tags uma when sibling bmtl uma_toolkit is present. BUILD_UMA=1 forces;
# BUILD_UMA=0 skips. Runtime default is ZEROLLAMA_UMA_SCHED=auto (gate if
# uma_daemon is up; else ungated MLX).
#
# Usage:
#   ./scripts/build/build_zerollama_mac.sh
#   BUILD_MLX=0 ./scripts/build/build_zerollama_mac.sh          # ggml only (fast)
#   BUILD_MLX=1 ./scripts/build/build_zerollama_mac.sh          # force MLX dylib rebuild
#   BUILD_UMA=0 ./scripts/build/build_zerollama_mac.sh          # no uma client
#   BUILD_LLAMA_SERVER=auto  (default) ensure patches + build llama-server when missing/stale
#   BUILD_LLAMA_SERVER=1     force llama-server rebuild
#   BUILD_LLAMA_SERVER=0     skip vendor patch check and llama-server build
#   BUILD_RUNTIME_KV_EXT=auto (default) rebuild runtime/kv/_kv_native when sources or libllama stale
#   BUILD_RUNTIME_KV_EXT=1   force _kv_native rebuild (needs vendor libllama)
#   BUILD_RUNTIME_KV_EXT=0   skip Python native ext (sidecar falls back to Python pool)
#   ./scripts/build/build_zerollama_mac.sh /path/to/output/binary
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP_DIR="${ROOT}/.build-stamps"
OUT="${1:-${ROOT}/zerollama}"
BUILD_RUNTIME_KV_EXT="${BUILD_RUNTIME_KV_EXT:-auto}"
if [[ -z "${VERSION:-}" ]]; then
  if git -C "${ROOT}" describe --tags --first-parent --abbrev=7 --long --dirty --always &>/dev/null; then
    VERSION="$(git -C "${ROOT}" describe --tags --first-parent --abbrev=7 --long --dirty --always | sed -e 's/^v//')"
  else
    VERSION="0.19.0"
  fi
fi
BUILD_MLX="${BUILD_MLX:-auto}"
BUILD_UMA="${BUILD_UMA:-auto}"

# shellcheck source=scripts/runtime/mac_cgo_env.sh
source "${ROOT}/scripts/runtime/mac_cgo_env.sh"
mac_cgo_env_warn_path
mac_cgo_env
mac_cgo_env_prefer_training_embed "${ROOT}"

if [[ ! -x "${ROOT}/.venv-training/bin/python" ]]; then
  echo ">>> note: .venv-training missing — training embed falls back to system python3-embed (3.9 on Xcode); run: MAC_SETUP_TRAINING=1 ./scripts/runtime/mac_setup.sh" >&2
fi

_mlx_dev_lib() {
  echo "${ROOT}/build/metal-v3/lib/ollama/mlx_metal_v3/libmlxc.dylib"
}

_mlx_dev_lib_v4() {
  echo "${ROOT}/build/metal-v4/lib/ollama/mlx_metal_v4/libmlxc.dylib"
}

_macos_major() {
  sw_vers -productVersion 2>/dev/null | cut -d. -f1
}

_should_build_mlx() {
  case "${BUILD_MLX}" in
    0) return 1 ;;
    1) return 0 ;;
    auto)
      local mlx_src="${OLLAMA_MLX_SOURCE:-${ROOT}/../mlx}"
      if [[ ! -d "${mlx_src}/.git" ]]; then
        echo ">>> BUILD_MLX=auto: skip MLX (no checkout at ${mlx_src})" >&2
        echo ">>>   safetensors: ./scripts/mlx/ensure_mlx_sources.sh --clone && BUILD_MLX=1 $0" >&2
        return 1
      fi
      if [[ -f "$(_mlx_dev_lib)" ]] && [[ "${MLX_FORCE:-0}" != "1" ]]; then
        local major
        major="$(_macos_major)"
        # WHY: v3-only skip left Tahoe hosts on Metal 3 metallib; loader prefers v4 when present.
        if [[ "${major:-0}" -ge 26 ]] && [[ ! -f "$(_mlx_dev_lib_v4)" ]]; then
          echo ">>> BUILD_MLX=auto: Metal v3 present, building v4 for macOS ${major}" >&2
          return 0
        fi
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

# shellcheck source=scripts/build/uma_toolkit.sh
source "${ROOT}/scripts/build/uma_toolkit.sh"

_uma_toolkit_root() {
  uma_toolkit_root
}

_should_build_uma() {
  case "${BUILD_UMA}" in
    0) return 1 ;;
    1) return 0 ;;
    auto)
      if uma_toolkit_present; then
        return 0
      fi
      echo ">>> BUILD_UMA=auto: skip uma (no toolkit at $(_uma_toolkit_root))" >&2
      return 1
      ;;
    *)
      echo "error: BUILD_UMA must be 0, 1, or auto (got ${BUILD_UMA})" >&2
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
  # WHY: ggml-org split server routes into libllama-server-impl.dylib; the
  # llama-server Mach-O is a thin wrapper (~30KB) that no longer embeds
  # "/kv/seq-copy". Probe the impl dylib (same as build_llama_server.sh).
  [[ -x "${LLAMA_SERVER_BIN}" && -f "${LLAMA_LIB}" ]] || return 1
  if grep -aqF 'kv/seq-copy' "${LLAMA_SERVER_BIN}" 2>/dev/null; then
    return 0
  fi
  local impl
  for impl in "$(dirname "${LLAMA_SERVER_BIN}")"/libllama-server-impl*; do
    if [[ -f "${impl}" ]] && grep -aqF 'kv/seq-copy' "${impl}" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

_llama_lib_has_kv_page_map() {
  # Prefer nm: `strings` can miss freshly-linked Mach-O exports under some PATH/toolchains.
  [[ -f "${LLAMA_LIB}" ]] || return 1
  if command -v nm >/dev/null 2>&1; then
    nm -gU "${LLAMA_LIB}" 2>/dev/null | grep -qE '[[:space:]]_?llama_memory_kv_page_map$' && return 0
  fi
  strings "${LLAMA_LIB}" 2>/dev/null | grep -qF 'llama_memory_kv_page_map'
}

_mac_sha_files() {
  shasum -a 256 "$@" 2>/dev/null | shasum -a 256 | awk '{print $1}'
}

_runtime_kv_native_fingerprint() {
  local lib="${LLAMA_LIB:-}"
  local lib_sha=""
  if [[ -f "${lib}" ]]; then
    lib_sha="$(shasum -a 256 "${lib}" 2>/dev/null | awk '{print $1}')"
  fi
  {
    _mac_sha_files \
      "${ROOT}/llama/llama.cpp/include/llama-kv-ext.h" \
      "${ROOT}/llama/llama.cpp/src/llama-memory-kv-ext.cpp" \
      "${ROOT}/runtime/setup.py" \
      "${ROOT}/runtime/native/kv_block_pool.c" \
      "${ROOT}/runtime/native/kv_tensor_probe.c" \
      "${ROOT}/runtime/native/kv_tensor_probe.h" \
      "${ROOT}/runtime/native/kv_page_bind_internal.h" \
      "${ROOT}/runtime/native/kv_decode_loop.c" \
      "${ROOT}/scripts/phase/phase15_runtime_kv_env.sh"
    echo "libllama=${lib_sha}"
  } | shasum -a 256 | awk '{print $1}'
}

_kv_native_ext_path() {
  find "${ROOT}/runtime/runtime/kv" -maxdepth 1 -name '_kv_native*.so' 2>/dev/null | head -1
}

_should_build_llama_server() {
  case "${BUILD_LLAMA_SERVER}" in
    0) return 1 ;;
    1) return 0 ;;
    auto|yes)
      if [[ ! -f "${VENDOR}/CMakeLists.txt" ]]; then
        echo ">>> BUILD_LLAMA_SERVER=${BUILD_LLAMA_SERVER}: vendor missing at ${VENDOR}" >&2
        echo ">>>   subprocess models (qwen3.6 MTP): ./scripts/vendor/rebase_vendor_unified.sh --apply --sync" >&2
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
      if ! _llama_lib_has_kv_page_map; then
        echo ">>> BUILD_LLAMA_SERVER=${BUILD_LLAMA_SERVER}: libllama missing llama_memory_kv_page_map (kv-ext v33)" >&2
        return 0
      fi
      if ! "${ROOT}/scripts/vendor/stage_llama_kv_ext_for_vendor.sh" "${VENDOR}" --check 2>/dev/null; then
        echo ">>> BUILD_LLAMA_SERVER=${BUILD_LLAMA_SERVER}: in-tree llama-kv-ext newer than vendor" >&2
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

_should_build_runtime_kv_ext() {
  case "${BUILD_RUNTIME_KV_EXT}" in
    0) return 1 ;;
    1) return 0 ;;
    auto|yes)
      if [[ ! -f "${VENDOR}/CMakeLists.txt" ]]; then
        echo ">>> BUILD_RUNTIME_KV_EXT=${BUILD_RUNTIME_KV_EXT}: skip (no vendor tree)" >&2
        return 1
      fi
      if [[ ! -f "${LLAMA_LIB}" ]]; then
        echo ">>> BUILD_RUNTIME_KV_EXT=${BUILD_RUNTIME_KV_EXT}: need libllama (BUILD_LLAMA_SERVER=auto will build)" >&2
        return 0
      fi
      local ext
      ext="$(_kv_native_ext_path)"
      local stamp="${STAMP_DIR}/runtime-kv-native.sha"
      local fp
      fp="$(_runtime_kv_native_fingerprint)"
      if [[ ! -f "${ext}" ]]; then
        echo ">>> BUILD_RUNTIME_KV_EXT=${BUILD_RUNTIME_KV_EXT}: _kv_native missing" >&2
        return 0
      fi
      if [[ ! -f "${stamp}" || "$(<"${stamp}")" != "${fp}" ]]; then
        echo ">>> BUILD_RUNTIME_KV_EXT=${BUILD_RUNTIME_KV_EXT}: sources or libllama changed" >&2
        return 0
      fi
      if [[ "${LLAMA_LIB}" -nt "${ext}" ]]; then
        echo ">>> BUILD_RUNTIME_KV_EXT=${BUILD_RUNTIME_KV_EXT}: libllama newer than _kv_native" >&2
        return 0
      fi
      echo ">>> BUILD_RUNTIME_KV_EXT=${BUILD_RUNTIME_KV_EXT}: _kv_native OK (cached)" >&2
      return 1
      ;;
    *)
      echo "error: BUILD_RUNTIME_KV_EXT must be 0, 1, or auto (got ${BUILD_RUNTIME_KV_EXT})" >&2
      exit 1
      ;;
  esac
}

_ensure_runtime_venv() {
  if [[ -x "${ROOT}/runtime/.venv/bin/python" ]]; then
    return 0
  fi
  if command -v uv >/dev/null 2>&1; then
    # shellcheck source=scripts/runtime/runtime_uv_venv.sh
    source "${ROOT}/scripts/runtime/runtime_uv_venv.sh"
    runtime_uv_venv
    return 0
  fi
  echo ">>> note: runtime/.venv missing — _kv_native build uses system python3" >&2
}

_build_runtime_kv_ext() {
  _ensure_runtime_venv
  # shellcheck source=scripts/phase/phase15_runtime_kv_env.sh
  source "${ROOT}/scripts/phase/phase15_runtime_kv_env.sh"
  export LLAMA_CPP_ROOT="${VENDOR}"
  export LLAMA_CPP_LIB="${LLAMA_LIB}"
  phase15_runtime_kv_env_apply
  phase15_runtime_kv_ext_build
  mkdir -p "${STAMP_DIR}"
  _runtime_kv_native_fingerprint > "${STAMP_DIR}/runtime-kv-native.sha"
  echo ">>> OK: runtime _kv_native linked to ${LLAMA_LIB}" >&2
}

_ensure_runtime_kv_ext() {
  BUILD_RUNTIME_KV_EXT="${BUILD_RUNTIME_KV_EXT:-auto}"
  _llama_vendor_paths
  if [[ ! -f "${VENDOR}/CMakeLists.txt" ]]; then
    return 0
  fi
  if _should_build_runtime_kv_ext; then
    if [[ ! -f "${LLAMA_LIB}" ]]; then
      if [[ "${BUILD_RUNTIME_KV_EXT}" == "1" ]]; then
        echo "error: BUILD_RUNTIME_KV_EXT=1 but ${LLAMA_LIB} missing — set BUILD_LLAMA_SERVER=auto" >&2
        exit 1
      fi
      echo ">>> warn: skip _kv_native — no ${LLAMA_LIB} (enable BUILD_LLAMA_SERVER=auto)" >&2
      return 0
    fi
    echo ">>> building runtime _kv_native (BUILD_RUNTIME_KV_EXT=${BUILD_RUNTIME_KV_EXT})" >&2
    _build_runtime_kv_ext
  fi
}

_ensure_and_build_llama_server() {
  BUILD_LLAMA_SERVER="${BUILD_LLAMA_SERVER:-auto}"
  _llama_vendor_paths

  if [[ ! -f "${VENDOR}/CMakeLists.txt" ]]; then
    case "${BUILD_LLAMA_SERVER}" in
      0) return 0 ;;
      1)
        echo "error: vendor missing at ${VENDOR}" >&2
        echo "  run: ./scripts/vendor/rebase_vendor_unified.sh --apply --sync" >&2
        exit 1
        ;;
      auto|yes)
        echo ">>> warn: vendor missing at ${VENDOR}; skipping llama-server (subprocess MTP unavailable)" >&2
        return 0
        ;;
    esac
  fi

  echo ">>> ensuring vendor Ollama patches + compat hooks" >&2
  "${ROOT}/scripts/vendor/ensure_llama_vendor_patches.sh" "${VENDOR}"
  "${ROOT}/scripts/vendor/stage_llama_kv_ext_for_vendor.sh" "${VENDOR}"

  if ! _llama_vendor_patched_ok; then
    echo "error: vendor still missing Ollama compat hooks after ensure_llama_vendor_patches.sh" >&2
    exit 1
  fi

  if _should_build_llama_server; then
    echo ">>> building llama-server (BUILD_LLAMA_SERVER=${BUILD_LLAMA_SERVER})" >&2
    LLAMA_CPP_ROOT="${VENDOR}" "${ROOT}/scripts/build/build_llama_server.sh"
  fi

  if [[ "${BUILD_LLAMA_SERVER}" != "0" ]]; then
    if ! _llama_server_binary_ok; then
      echo "error: llama-server build finished but binary probe failed" >&2
      exit 1
    fi
    if ! _llama_lib_has_kv_page_map; then
      echo "error: libllama missing llama_memory_kv_page_map after build — check stage_llama_kv_ext_for_vendor.sh" >&2
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
METAL_DIR="${ROOT}/ml/backend/ggml/ggml/src/ggml-metal"
echo ">>> generating per-kind Metal embeds (kernels + eliza-shipped kinds)" >&2
bash "${ROOT}/scripts/build/gen_ggml_metal_embed.sh"
if ! grep -q 'GGML_METAL_LIBS' "${METAL_DIR}/ggml-metal-device.m"; then
  echo "error: ggml-metal-device.m is missing GGML_METAL_LIBS (per-kind embed loader)" >&2
  exit 1
fi
# Force as to re-.incbin if only .metal payloads changed.
touch "${METAL_DIR}/ggml-metal-embed.s"

if _should_build_mlx; then
  echo ">>> building MLX dylibs (BUILD_MLX=${BUILD_MLX})" >&2
  bash "${ROOT}/scripts/build/build_mlx_dylibs_mac.sh"
fi

_ensure_and_build_llama_server

_ensure_runtime_kv_ext

if [[ "$(uname -s)" == Darwin && -f "${ROOT}/scripts/vendor/restore_ane_hook_intree.sh" ]]; then
  echo ">>> syncing in-tree ANE hook sources (CGO llama-common)" >&2
  "${ROOT}/scripts/vendor/restore_ane_hook_intree.sh"
fi

# CGO llama.cpp/common needs nlohmann/miniaudio/stb under llama/llama.cpp/vendor.
# b10488 mtmd also needs vendor/hash (sha256.c) via llama/cgo_vendor_hash.cpp.
# Repo .gitignore matches any vendor/, so pin syncs leave these missing unless staged.
_ensure_llama_cgo_vendor_headers() {
  local dest="${ROOT}/llama/llama.cpp/vendor"
  local need=0
  [[ -f "${dest}/nlohmann/json.hpp" && -f "${dest}/miniaudio/miniaudio.h" && -f "${dest}/stb/stb_image.h" ]] || need=1
  [[ -f "${dest}/hash/sha256/sha256.c" ]] || need=1
  if [[ "${need}" -eq 0 ]]; then
    return 0
  fi
  local src="" d
  local pin
  pin="$(tr -d '[:space:]' <"${ROOT}/LLAMA_CPP_COMMIT" 2>/dev/null || true)"
  for d in \
    "${ROOT}/vendor/llama-cpp-${pin}/vendor" \
    "${ROOT}/vendor/llama-cpp-b10488/vendor" \
    "${ROOT}/vendor/llama-cpp-${pin:0:8}/vendor" \
    "${ROOT}"/vendor/llama-cpp-*/vendor \
    "${ROOT}/../llama.cpp/vendor"
  do
    if [[ -f "${d}/nlohmann/json.hpp" || -f "${d}/hash/sha256/sha256.c" ]]; then
      src="${d}"
      [[ -f "${d}/nlohmann/json.hpp" ]] && break
    fi
  done
  if [[ -z "${src}" ]]; then
    echo ">>> warn: missing llama.cpp vendor extras under ${dest}; CGO build may fail (sync vendor pin)" >&2
    return 0
  fi
  echo ">>> restoring CGO llama vendor headers from ${src}" >&2
  mkdir -p "${dest}"
  for name in nlohmann miniaudio stb hash; do
    if [[ -d "${src}/${name}" ]]; then
      rm -rf "${dest:?}/${name}"
      cp -R "${src}/${name}" "${dest}/${name}"
    fi
  done
}
_ensure_llama_cgo_vendor_headers

GO_TAGS=()
if _should_build_uma; then
  echo ">>> building uma broker client (BUILD_UMA=${BUILD_UMA})" >&2
  make -C "${ROOT}/x/uma" "BMTL_UMA_TOOLKIT=$(_uma_toolkit_root)"
  # cgo may not re-link when only libuma_embed.a changes
  touch "${ROOT}/x/uma/uma_darwin.go"
  GO_TAGS+=(uma)
fi

GO_BUILD_ARGS=(-ldflags="-s -w -X=github.com/ollama/ollama/version.Version=${VERSION}" -o "${OUT}")
if ((${#GO_TAGS[@]})); then
  GO_BUILD_ARGS+=(-tags "$(IFS=,; echo "${GO_TAGS[*]}")")
fi
GOFLAGS=-mod=mod go build "${GO_BUILD_ARGS[@]}" .
if ((${#GO_TAGS[@]})) && ! strings "${OUT}" | grep -q 'cum_leases'; then
  echo ">>> uma: cgo miss — rebuilding with -a" >&2
  touch "${ROOT}/x/uma/uma_darwin.go"
  GOFLAGS=-mod=mod go build -a "${GO_BUILD_ARGS[@]}" .
fi
echo ">>> wrote ${OUT} (version ${VERSION}${GO_TAGS:+ tags=${GO_TAGS[*]}})" >&2
