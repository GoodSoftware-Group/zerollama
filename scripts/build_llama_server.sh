#!/usr/bin/env bash
# Build llama-server from pinned llama.cpp (see runtime/LLAMA_CPP_PIN.md).
#
# Unified source: vendor/llama-cpp-<pin> (elizaOS + Ollama patches). Falls back to
# ../llama.cpp sibling when LLAMA_CPP_ROOT is unset and vendor is missing.
#
# LLAMA_CPP_ROOT overrides both. Why vendor default: patches (kv-ext, compat hooks,
# GPU discovery) must match in-process ggml — bare eliza sibling is not sufficient.
#
# Why validate nvcc: CMAKE may find headers under cuda-12.8 while CUDACXX points at a
# missing cuda-13/bin/nvcc. RTX 5080: CMAKE_CUDA_ARCHITECTURES=120-real (see docs/testing-smoke.md).
# macOS (M3): GGML_METAL=ON, GGML_CUDA=OFF — produces libllama.dylib + llama-server.
set -euo pipefail

_ZEROLLAMA_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
_FETCH_HEAD="$(grep '^FETCH_HEAD=' "${_ZEROLLAMA_ROOT}/Makefile.sync" 2>/dev/null | cut -d= -f2 || echo c84b3020)"
_VENDOR_ROOT="${_ZEROLLAMA_ROOT}/vendor/llama-cpp-${_FETCH_HEAD}"
_SIBLING_ROOT="${_ZEROLLAMA_ROOT}/../llama.cpp"

_resolve_llama_cpp_root() {
  if [[ -n "${LLAMA_CPP_ROOT:-}" ]]; then
    echo "${LLAMA_CPP_ROOT}"
    return 0
  fi
  if [[ -f "${_VENDOR_ROOT}/CMakeLists.txt" ]]; then
    echo "${_VENDOR_ROOT}"
    return 0
  fi
  echo "${_SIBLING_ROOT}"
}

ROOT="$(_resolve_llama_cpp_root)"
BUILD="${ROOT}/build"
BUILD_LOCK="${ROOT}/.zerollama_llama_server_build.lock.d"

_acquire_llama_server_build_lock() {
  if pgrep -f "cmake --build ${BUILD}" >/dev/null 2>&1 \
    || pgrep -f "make.*${ROOT}/build" >/dev/null 2>&1; then
    echo "error: llama-server build still running for ${ROOT}" >&2
    echo "  wait for it to finish, or: pkill -f 'cmake --build ${BUILD}'" >&2
    exit 1
  fi
  if ! mkdir "${BUILD_LOCK}" 2>/dev/null; then
    echo "error: llama-server build lock held for ${ROOT}" >&2
    echo "  wait for the other build, or if stale: rm -rf ${BUILD_LOCK}" >&2
    exit 1
  fi
  trap '_release_llama_server_build_lock' EXIT
}

_release_llama_server_build_lock() {
  rmdir "${BUILD_LOCK}" 2>/dev/null || true
}

_build_jobs() {
  local n="${LLAMA_BUILD_JOBS:-}"
  if [[ -n "${n}" ]]; then
    echo "${n}"
    return 0
  fi
  n="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
  # High -j races rm -rf + link on the same tree (missing bin/*.dylib, *.o.d).
  if [[ "${n}" -gt 8 ]]; then
    n=8
  fi
  echo "${n}"
}

# shellcheck source=scripts/ensure_llama_cpp_sibling.sh
source "${_ZEROLLAMA_ROOT}/scripts/ensure_llama_cpp_sibling.sh"

if [[ ! -f "${ROOT}/CMakeLists.txt" ]]; then
  if [[ "${ROOT}" == "${_VENDOR_ROOT}" ]]; then
    echo "patched vendor missing at ${ROOT}; run ./scripts/rebase_vendor_unified.sh --sync" >&2
    exit 1
  fi
  ensure_llama_cpp_sibling || {
    echo "llama.cpp not found at ${ROOT}" >&2
    exit 1
  }
  ROOT="$(_resolve_llama_cpp_root)"
  BUILD="${ROOT}/build"
fi
if [[ -d "${ROOT}/.git" && "${ROOT}" != "${_VENDOR_ROOT}" ]]; then
  ensure_llama_cpp_at_pin "${ROOT}"
fi

# Patched vendor builds need Ollama patch commits + compat loader hooks + staged sources.
if [[ "${ROOT}" == "${_VENDOR_ROOT}" || "${ROOT}" == "${_VENDOR_ROOT}/" ]]; then
  "${_ZEROLLAMA_ROOT}/scripts/ensure_llama_vendor_patches.sh" "${ROOT}"
fi

# Darwin sibling fallback: re-apply ANE hook after pin checkout (vendor gets 0018 via git am).
if [[ "$(uname -s)" == Darwin && "${ZEROLLAMA_SKIP_ANE_HOOK_SYNC:-0}" != "1" ]]; then
  if [[ "${ROOT}" != "${_VENDOR_ROOT}" && "${ROOT}" != "${_VENDOR_ROOT}/" ]]; then
    if [[ -f "${_ZEROLLAMA_ROOT}/scripts/sync_ane_hook_to_llama_cpp.sh" ]]; then
      echo ">>> sync ANE draft hook → ${ROOT}" >&2
      LLAMA_CPP_ROOT="${ROOT}" "${_ZEROLLAMA_ROOT}/scripts/sync_ane_hook_to_llama_cpp.sh"
    fi
  fi
fi

_probe_ollama_compat_loader() {
  local vendor_root="$1"
  if [[ "${vendor_root}" != "${_VENDOR_ROOT}" && "${vendor_root}" != "${_VENDOR_ROOT}/" ]]; then
    return 0
  fi
  if grep -q 'llama_ollama_compat::translate_metadata' "${vendor_root}/src/llama-model-loader.cpp"; then
    echo "OK: vendor loader calls llama_ollama_compat::translate_metadata"
  else
    echo "error: vendor loader missing Ollama compat hooks — run ./scripts/ensure_llama_vendor_patches.sh" >&2
    exit 1
  fi
}

if [[ "$(uname -s)" != "Darwin" ]]; then
  bash "${_ZEROLLAMA_ROOT}/scripts/patch_vendor_linux_ane_hook.sh" "${ROOT}"
  if [[ -f "${_ZEROLLAMA_ROOT}/llama/llama.cpp/common/ane_draft_hook.cpp" ]]; then
    bash "${_ZEROLLAMA_ROOT}/scripts/patch_vendor_linux_ane_hook.sh" "${_ZEROLLAMA_ROOT}/llama/llama.cpp"
  fi
fi

_probe_llama_server_capabilities() {
  local bin="$1"
  local help_text
  local libdir
  libdir="$(dirname "${bin}")"
  help_text="$(LD_LIBRARY_PATH="${libdir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" "${bin}" --help 2>&1 || true)"
  if echo "${help_text}" | grep -qE 'qjl1_256|q4_polar|tbq3_0|tbq4_0'; then
    echo "OK: ${bin} advertises eliza KV cache types"
  else
    echo "warn: ${bin} missing fork KV types in --help (wrong ref?)" >&2
  fi
  if echo "${help_text}" | grep -q 'checkpoint-every-n-tokens'; then
    echo "OK: ${bin} advertises --checkpoint-every-n-tokens"
  else
    echo "warn: ${bin} missing --checkpoint-every-n-tokens" >&2
  fi
  if echo "${help_text}" | grep -qE 'ctx-checkpoints|swa-checkpoints'; then
    echo "OK: ${bin} advertises ctx/swa checkpoints"
  fi
}

_probe_seq_copy_route() {
  local bin="$1"
  local root="$2"
  local has_route=0
  if grep -aqF 'kv/seq-copy' "${bin}" 2>/dev/null; then
    has_route=1
  elif strings "${bin}" 2>/dev/null | grep -qF 'kv/seq-copy'; then
    has_route=1
  fi
  # WHY: Radix cross-slot seed requires patch 0017 on vendor tree only.
  if [[ "${root}" != "${_VENDOR_ROOT}" && "${root}" != "${_VENDOR_ROOT}/" ]]; then
    if [[ "${has_route}" -eq 1 ]]; then
      echo "OK: ${bin} embeds /kv/seq-copy (non-vendor build)"
    else
      echo "warn: ${bin} missing /kv/seq-copy (expected for bare sibling builds)" >&2
    fi
    return 0
  fi
  if [[ "${has_route}" -eq 1 ]]; then
    echo "OK: ${bin} embeds POST /kv/seq-copy (patch 0017)"
  else
    echo "error: ${bin} missing /kv/seq-copy — rebuild from patched vendor" >&2
    echo "  ./scripts/rebase_vendor_unified.sh --apply --sync && ./scripts/build_llama_server.sh" >&2
    exit 1
  fi
}

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
  _acquire_llama_server_build_lock
  rm -rf "${BUILD}"
  cmake -S "${ROOT}" -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DGGML_METAL=ON \
    -DGGML_CUDA=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_WEBUI="${LLAMA_BUILD_WEBUI:-OFF}"
  cmake --build "${BUILD}" --target llama-server -j"$(_build_jobs)" || {
    echo "error: llama-server build failed; cleaning ${BUILD}" >&2
    rm -rf "${BUILD}"
    exit 1
  }
  BIN="${BUILD}/bin/llama-server"
  LIB="${BUILD}/bin/libllama.dylib"
  ANE_REPO="${ANE_REPO:-${HOME}/Sites/inference/ane}"
  ANE_BRIDGE="${ANE_REPO}/bridge/libane_bridge.dylib"
  if [[ -f "${ANE_BRIDGE}" ]]; then
    cp -f "${ANE_BRIDGE}" "${BUILD}/bin/"
    # Why libllama-common: ane_draft_session links ane_bridge; dyld loads bridge from common.dylib.
    for _ane_fix in "${BUILD}/bin/libllama-common"*.dylib "${BIN}"; do
      if [[ -f "${_ane_fix}" ]]; then
        install_name_tool -change libane_bridge.dylib @loader_path/libane_bridge.dylib "${_ane_fix}" 2>/dev/null || true
      fi
    done
    echo "OK: copied libane_bridge.dylib to ${BUILD}/bin/"
  fi
  if [[ -x "${BIN}" && -f "${LIB}" ]]; then
    echo "OK: ${BIN}"
    echo "OK: ${LIB}"
    "${BIN}" --version 2>/dev/null || true
    _probe_llama_server_capabilities "${BIN}"
    _probe_ollama_compat_loader "${ROOT}"
    _probe_seq_copy_route "${BIN}" "${ROOT}"
    git -C "${ROOT}" rev-parse HEAD > "${BUILD}/.zerollama_vendor_rev" 2>/dev/null || true
  else
    echo "Build finished but ${BIN} or ${LIB} missing" >&2
    exit 1
  fi
  exit 0
fi

export LLAMA_BUILD_WEBUI="${LLAMA_BUILD_WEBUI:-OFF}"

if [[ "${GGML_VULKAN:-}" == "ON" || "${GGML_VULKAN:-}" == "1" ]]; then
  echo "Building llama-server in ${ROOT} (Vulkan)"
  rm -rf "${BUILD}"
  cmake -S "${ROOT}" -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON \
    -DGGML_CUDA=OFF \
    -DGGML_METAL=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_CURL=ON \
    -DLLAMA_BUILD_WEBUI="${LLAMA_BUILD_WEBUI}"
  cmake --build "${BUILD}" --target llama-server -j"$(nproc)"
  BIN="${BUILD}/bin/llama-server"
  if [[ -x "${BIN}" ]]; then
    echo "OK: ${BIN}"
    "${BIN}" --version 2>/dev/null || true
    _probe_llama_server_capabilities "${BIN}"
    if [[ "${ROOT}" == "${_VENDOR_ROOT}" || "${ROOT}" == "${_VENDOR_ROOT}/" ]]; then
      _probe_seq_copy_route "${BIN}" "${ROOT}"
    fi
  else
    echo "Build finished but ${BIN} missing" >&2
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

_acquire_llama_server_build_lock
rm -rf "${BUILD}"
# WHY LLAMA_BUILD_WEBUI: eliza fork defaults ON; headless Linux builds fail without WebUI assets.
# WHY GGML_CUDA_GRAPHS: L3 prefix cache clears KV slots; zerollama calls
# llama_context_cuda_graph_invalidate (in-process) or POST /cuda-graph/invalidate
# (subprocess llama-server) to drop stale captured graphs on CUDA.
cmake -S "${ROOT}" -B "${BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA="${GGML_CUDA:-ON}" \
  -DGGML_CUDA_GRAPHS=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
  -DLLAMA_CURL=ON \
  -DLLAMA_BUILD_WEBUI="${LLAMA_BUILD_WEBUI:-ON}"

cmake --build "${BUILD}" --target llama-server -j"$(_build_jobs)" || {
  echo "error: llama-server build failed; cleaning ${BUILD}" >&2
  rm -rf "${BUILD}"
  exit 1
}

BIN="${BUILD}/bin/llama-server"
if [[ -x "${BIN}" ]]; then
  echo "OK: ${BIN}"
  "${BIN}" --version 2>/dev/null || true
  _probe_llama_server_capabilities "${BIN}"
  _probe_ollama_compat_loader "${ROOT}"
  _probe_seq_copy_route "${BIN}" "${ROOT}"
else
  echo "Build finished but ${BIN} missing" >&2
  exit 1
fi
