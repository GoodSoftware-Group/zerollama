#!/usr/bin/env bash
# Install CUDA llama-server + ggml libs built by ./scripts/build/build_llama_server.sh
# into the Ollama layout under /usr/local/lib/ollama (dual-GPU / RTX 4090 hosts).
#
# Usage:
#   LLAMA_BUILD_WEBUI=OFF ./scripts/build/build_llama_server.sh
#   sudo ./scripts/vendor/install_cuda_llama_server.sh
#
# Env:
#   LLAMA_CPP_ROOT     — source tree (default ../llama.cpp)
#   OLLAMA_LIB_DIR     — install root (default /usr/local/lib/ollama)
#   OLLAMA_CUDA_VARIANT — cuda subdir (default cuda_v12)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
_FETCH_HEAD="$(grep '^FETCH_HEAD=' "${ROOT}/Makefile.sync" 2>/dev/null | cut -d= -f2 || echo c84b3020)"
_VENDOR_ROOT="${ROOT}/vendor/llama-cpp-${_FETCH_HEAD}"
_SIBLING_ROOT="${ROOT}/../llama.cpp"

_resolve_build_bin() {
  if [[ -n "${LLAMA_CPP_ROOT:-}" ]]; then
    echo "${LLAMA_CPP_ROOT}/build/bin"
    return 0
  fi
  if [[ -x "${_VENDOR_ROOT}/build/bin/llama-server" ]]; then
    echo "${_VENDOR_ROOT}/build/bin"
    return 0
  fi
  if [[ -x "${_SIBLING_ROOT}/build/bin/llama-server" ]]; then
    echo "${_SIBLING_ROOT}/build/bin"
    return 0
  fi
  echo "${_SIBLING_ROOT}/build/bin"
}

BUILD_DIR="$(_resolve_build_bin)"
INSTALL_DIR="${OLLAMA_LIB_DIR:-/usr/local/lib/ollama}"
CUDA_DIR="${INSTALL_DIR}/${OLLAMA_CUDA_VARIANT:-cuda_v12}"
CUDA_SUB="${OLLAMA_CUDA_VARIANT:-cuda_v12}"

if [[ ! -x "${BUILD_DIR}/llama-server" ]]; then
  echo "install_cuda: missing ${BUILD_DIR}/llama-server — run:" >&2
  echo "  ./scripts/vendor/build_llama_server_container.sh" >&2
  exit 1
fi

if [[ "$(id -u)" -ne 0 ]]; then
  echo "install_cuda: re-run with sudo" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP="${INSTALL_DIR}.bak-${STAMP}"
echo ">>> backup ${INSTALL_DIR} → ${BACKUP}"
cp -a "${INSTALL_DIR}" "${BACKUP}"

mkdir -p "${INSTALL_DIR}" "${CUDA_DIR}"

echo ">>> install llama-server from ${BUILD_DIR}"
install -m 755 "${BUILD_DIR}/llama-server" "${INSTALL_DIR}/llama-server"

# Copy all shared libs from build/bin (versioned symlinks + real files).
shopt -s nullglob
for f in "${BUILD_DIR}"/*.so "${BUILD_DIR}"/*.so.*; do
  base="$(basename "${f}")"
  case "${base}" in
    libggml-cuda.so*) continue ;;
  esac
  install -m 755 "${f}" "${INSTALL_DIR}/${base}"
done
for f in "${BUILD_DIR}"/libggml-cuda.so*; do
  install -m 755 "${f}" "${CUDA_DIR}/$(basename "${f}")"
done
# Tensor-parallel builds may link NCCL; prefer Ollama cuda_v12 bundle, then cuda-12.6 targets.
for _nccl in \
  "${CUDA_DIR}/libnccl.so.2" \
  /usr/local/cuda-12.6/targets/x86_64-linux/lib/libnccl.so.2 \
  /usr/local/cuda/targets/x86_64-linux/lib/libnccl.so.2; do
  if [[ -f "${_nccl}" && ! -f "${CUDA_DIR}/libnccl.so.2" ]]; then
    install -m 755 "${_nccl}" "${CUDA_DIR}/libnccl.so.2"
    echo ">>> installed libnccl.so.2 from ${_nccl}"
    break
  fi
done
shopt -u nullglob

_set_install_rpath() {
  local f="$1"
  local rpath='$ORIGIN:$ORIGIN/'"${CUDA_SUB}"
  if command -v patchelf >/dev/null 2>&1; then
    patchelf --set-rpath "${rpath}" "${f}" 2>/dev/null || true
  elif command -v chrpath >/dev/null 2>&1; then
    chrpath -r "${rpath}" "${f}" 2>/dev/null || true
  fi
}

echo ">>> set RPATH (\$ORIGIN:\$ORIGIN/${CUDA_SUB}) on installed libs"
_set_install_rpath "${INSTALL_DIR}/llama-server"
for f in "${INSTALL_DIR}"/*.so "${INSTALL_DIR}"/*.so.* "${CUDA_DIR}"/*.so "${CUDA_DIR}"/*.so.*; do
  [[ -f "${f}" ]] || continue
  _set_install_rpath "${f}"
done
if ! command -v patchelf >/dev/null 2>&1 && ! command -v chrpath >/dev/null 2>&1; then
  echo "warn: patchelf/chrpath missing — verify readelf -d llama-server | grep RUNPATH" >&2
fi

# Preserve Ollama-bundled cublas/cudart in cuda_v12 if present.
chmod -R a+rX "${INSTALL_DIR}"

echo ">>> verify fork KV types + GPU backend"
export LD_LIBRARY_PATH="${INSTALL_DIR}:${CUDA_DIR}"
export GGML_BACKEND_PATH="${CUDA_DIR}/libggml-cuda.so"
HELP="$("${INSTALL_DIR}/llama-server" --help 2>&1 || true)"
if echo "${HELP}" | grep -qE 'qjl1_256|q4_polar|tbq3_0|tbq4_0'; then
  echo "OK: fork KV cache types in --help"
else
  echo "warn: fork KV types not in --help (stock build?)" >&2
fi
if echo "${HELP}" | grep -q 'ctx-checkpoints'; then
  echo "OK: ctx-checkpoints advertised"
fi
CUDA_LIB=""
for cand in "${CUDA_DIR}/libggml-cuda.so.0.12.0" "${CUDA_DIR}/libggml-cuda.so.0" "${CUDA_DIR}/libggml-cuda.so"; do
  [[ -e "${cand}" ]] && CUDA_LIB="$(readlink -f "${cand}")" && break
done
if [[ -n "${CUDA_LIB}" && -f "${CUDA_LIB}" ]]; then
  _installed_cuda_has_symbol() {
    local sym="$1"
    grep -Fq "${sym}" < <(strings "${CUDA_LIB}" 2>/dev/null) && return 0
    command -v nm >/dev/null 2>&1 || return 1
    nm --demangle "${CUDA_LIB}" 2>/dev/null | grep -Fq "${sym}" && return 0
    return 1
  }
  if ! _installed_cuda_has_symbol 'ggml_cuda_op_set_rows'; then
    echo "error: ${CUDA_LIB} missing ggml_cuda_op_set_rows" >&2
    exit 1
  fi
  echo "OK: libggml-cuda SET_ROWS dispatch"
  if _installed_cuda_has_symbol 'fused_attn_qjl_polar_cuda'; then
    echo "OK: libggml-cuda fused QJL attn"
  else
    echo "error: ${CUDA_LIB} missing fused QJL attn symbols" >&2
    exit 1
  fi
else
  echo "warn: libggml-cuda not found under ${CUDA_DIR}" >&2
fi
RPATH="$(readelf -d "${INSTALL_DIR}/llama-server" 2>/dev/null | grep -E 'RUNPATH|RPATH' || true)"
if echo "${RPATH}" | grep -q '\$ORIGIN'; then
  echo "OK: llama-server RPATH uses \$ORIGIN"
else
  echo "warn: llama-server may not resolve libs from ${INSTALL_DIR} (RPATH: ${RPATH:-none})" >&2
fi
"${INSTALL_DIR}/llama-server" --version 2>&1 | head -1 || true

echo ">>> install OK: ${INSTALL_DIR}/llama-server"
echo ">>> backup kept at ${BACKUP}"
echo ">>> restart: systemctl restart zerollama-runtime ollama"
