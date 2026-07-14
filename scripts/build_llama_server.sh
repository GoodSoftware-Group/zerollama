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

_canonical_path() {
  readlink -f "$1" 2>/dev/null || realpath "$1" 2>/dev/null || (cd "$1" && pwd)
}

# Container mounts vendor at /llama.cpp while _VENDOR_ROOT is /zerollama/vendor/... — same tree.
_is_vendor_root() {
  local root="$1"
  [[ -f "${root}/CMakeLists.txt" ]] || return 1
  local a b
  a="$(_canonical_path "${root}")"
  b="$(_canonical_path "${_VENDOR_ROOT}")"
  [[ "${a}" == "${b}" ]] && return 0
  # Podman bind-mount alias: same inode, different path.
  if [[ -f "${_VENDOR_ROOT}/CMakeLists.txt" ]]; then
    local ia ib
    ia="$(stat -c '%d:%i' "${root}/CMakeLists.txt" 2>/dev/null || stat -f '%d:%i' "${root}/CMakeLists.txt")"
    ib="$(stat -c '%d:%i' "${_VENDOR_ROOT}/CMakeLists.txt" 2>/dev/null || stat -f '%d:%i' "${_VENDOR_ROOT}/CMakeLists.txt")"
    [[ "${ia}" == "${ib}" ]] && return 0
  fi
  return 1
}

ROOT="$(_resolve_llama_cpp_root)"
BUILD="${ZEROLLAMA_BUILD_DIR:-${ROOT}/build}"
BUILD_LOCK="${BUILD}/.zerollama_llama_server_build.lock.d"
if [[ "${BUILD}" != "${ROOT}/build" ]]; then
  BUILD_LOCK="${ROOT}/.zerollama_llama_server_build.lock.d"
fi

_acquire_llama_server_build_lock() {
  if pgrep -f "cmake --build ${BUILD}" >/dev/null 2>&1; then
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
  BUILD="${ZEROLLAMA_BUILD_DIR:-${ROOT}/build}"
fi
if [[ -d "${ROOT}/.git" ]] && ! _is_vendor_root "${ROOT}" \
    && [[ "${ZEROLLAMA_SKIP_PIN_CHECKOUT:-0}" != "1" ]]; then
  ensure_llama_cpp_at_pin "${ROOT}"
fi

# Patched vendor: ensure Ollama + CUDA fork patches (skip when ZEROLLAMA_SKIP_VENDOR_APPLY=1).
if _is_vendor_root "${ROOT}"; then
  if [[ "${ZEROLLAMA_SKIP_VENDOR_APPLY:-0}" != "1" ]]; then
    "${_ZEROLLAMA_ROOT}/scripts/apply_llama_vendor_patches.sh" "${ROOT}"
  fi
fi

# Darwin sibling fallback: re-apply ANE hook after pin checkout (vendor gets 0018 via git am).
if [[ "$(uname -s)" == Darwin && "${ZEROLLAMA_SKIP_ANE_HOOK_SYNC:-0}" != "1" ]]; then
  if ! _is_vendor_root "${ROOT}"; then
    if [[ -f "${_ZEROLLAMA_ROOT}/scripts/sync_ane_hook_to_llama_cpp.sh" ]]; then
      echo ">>> sync ANE draft hook → ${ROOT}" >&2
      LLAMA_CPP_ROOT="${ROOT}" "${_ZEROLLAMA_ROOT}/scripts/sync_ane_hook_to_llama_cpp.sh"
    fi
  fi
fi

_probe_ollama_compat_loader() {
  local vendor_root="$1"
  if ! _is_vendor_root "${vendor_root}"; then
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
  if [[ -f "${ROOT}/common/ane_draft_hook.cpp" ]]; then
    bash "${_ZEROLLAMA_ROOT}/scripts/patch_vendor_linux_ane_hook.sh" "${ROOT}"
  fi
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

_probe_cuda_fork_cuda_symbols() {
  local bindir="$1"
  local cuda_lib=""
  for cand in \
    "${bindir}/libggml-cuda.so.0.12.0" \
    "${bindir}/libggml-cuda.so.0" \
    "${bindir}/libggml-cuda.so"; do
    if [[ -e "${cand}" ]]; then
      cuda_lib="$(readlink -f "${cand}")"
      break
    fi
  done
  if [[ -z "${cuda_lib}" || ! -f "${cuda_lib}" ]]; then
    echo "error: libggml-cuda not found under ${bindir}" >&2
    return 1
  fi
  _cuda_lib_has_symbol() {
    local sym="$1"
    nm -D --demangle "${cuda_lib}" 2>/dev/null | grep -Fq "${sym}" && return 0
    nm --demangle "${cuda_lib}" 2>/dev/null | grep -Fq "${sym}" && return 0
    strings "${cuda_lib}" 2>/dev/null | grep -Fq "${sym}" && return 0
    return 1
  }
  if ! _cuda_lib_has_symbol 'ggml_cuda_op_set_rows'; then
    echo "error: ${cuda_lib} missing ggml_cuda_op_set_rows (fork KV load will abort)" >&2
    return 1
  fi
  echo "OK: libggml-cuda SET_ROWS dispatch present"
  if _cuda_lib_has_symbol 'fused_attn_qjl_polar_cuda'; then
    echo "OK: libggml-cuda fused QJL attn symbols present"
  else
    echo "error: ${cuda_lib} missing fused QJL attn (build with -DGGML_CUDA_FUSED_ATTN_QJL=ON)" >&2
    return 1
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
  else
    # WHY: ggml-org thin llama-server wrapper; routes live in libllama-server-impl.
    local impl
    for impl in "$(dirname "${bin}")"/libllama-server-impl*; do
      if [[ -f "${impl}" ]] && { grep -aqF 'kv/seq-copy' "${impl}" 2>/dev/null \
          || strings "${impl}" 2>/dev/null | grep -qF 'kv/seq-copy'; }; then
        has_route=1
        break
      fi
    done
  fi
  # WHY: Radix cross-slot seed requires patch 0017 on vendor tree only.
  if ! _is_vendor_root "${root}"; then
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
    if _is_vendor_root "${ROOT}"; then
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
    /usr/bin
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
# WHY stubs: link llama-server in devel containers / CI without libcuda.so.1 on LD path.
CUDA_STUBS=""
for _stub in /usr/local/cuda/lib64/stubs /usr/local/cuda/targets/x86_64-linux/lib/stubs; do
  if [[ -f "${_stub}/libcuda.so" ]]; then
    CUDA_STUBS="${_stub}"
    break
  fi
done
CMAKE_EXTRA=()
if [[ -n "${CUDA_STUBS}" ]]; then
  CMAKE_EXTRA+=("-DCMAKE_EXE_LINKER_FLAGS=-L${CUDA_STUBS} -Wl,-rpath-link,${CUDA_STUBS} -lcuda")
fi
# Installed layout: libs in $ORIGIN, CUDA backend in $ORIGIN/cuda_v12 (see install_cuda_llama_server.sh).
CMAKE_EXTRA+=(
  "-DCMAKE_BUILD_RPATH=\$ORIGIN:\$ORIGIN/cuda_v12"
  "-DCMAKE_INSTALL_RPATH=\$ORIGIN:\$ORIGIN/cuda_v12"
  "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON"
)
# WHY LLAMA_BUILD_UI/WEBUI: ggml-org 8f114a9b+ uses LLAMA_BUILD_UI (WEBUI deprecated);
# headless Linux builds fail without HF UI assets when left ON.
# WHY LLAMA_USE_PREBUILT_UI=OFF with UI off: partial HF dist (missing loading.html)
# makes llama-ui-embed abort; empty stub embed is fine for llama-server API use.
# WHY GGML_CUDA_GRAPHS: L3 prefix cache clears KV slots; zerollama calls
# llama_context_cuda_graph_invalidate (in-process) or POST /cuda-graph/invalidate
# (subprocess llama-server) to drop stale captured graphs on CUDA.
_LLAMA_UI="${LLAMA_BUILD_UI:-${LLAMA_BUILD_WEBUI:-OFF}}"
_LLAMA_PREBUILT_UI="${LLAMA_USE_PREBUILT_UI:-}"
if [[ -z "${_LLAMA_PREBUILT_UI}" ]]; then
  if [[ "${_LLAMA_UI}" == "ON" || "${_LLAMA_UI}" == "1" ]]; then
    _LLAMA_PREBUILT_UI=ON
  else
    _LLAMA_PREBUILT_UI=OFF
  fi
fi
cmake -S "${ROOT}" -B "${BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA="${GGML_CUDA:-ON}" \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_FUSED_ATTN_QJL=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
  -DLLAMA_CURL=ON \
  -DLLAMA_BUILD_UI="${_LLAMA_UI}" \
  -DLLAMA_BUILD_WEBUI="${_LLAMA_UI}" \
  -DLLAMA_USE_PREBUILT_UI="${_LLAMA_PREBUILT_UI}" \
  -DLLAMA_USE_PREBUILT_WEBUI="${_LLAMA_PREBUILT_UI}" \
  "${CMAKE_EXTRA[@]}"

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
  if [[ "${ZEROLLAMA_SKIP_BUILD_PROBES:-0}" != "1" ]]; then
    _probe_cuda_fork_cuda_symbols "$(dirname "${BIN}")"
    _probe_ollama_compat_loader "${ROOT}"
    _probe_seq_copy_route "${BIN}" "${ROOT}"
  fi
else
  echo "Build finished but ${BIN} missing" >&2
  exit 1
fi
