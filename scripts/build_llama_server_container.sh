#!/usr/bin/env bash
# Build llama-server inside NVIDIA CUDA devel image (avoids host CUDA/CCCL skew).
#
#   ./scripts/build_llama_server_container.sh
#
# Output: vendor/llama-cpp-<pin>/build/bin/llama-server (or sibling when vendor missing).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_FETCH_HEAD="$(grep '^FETCH_HEAD=' "${ROOT}/Makefile.sync" 2>/dev/null | cut -d= -f2 || echo c84b3020)"
_VENDOR_ROOT="${ROOT}/vendor/llama-cpp-${_FETCH_HEAD}"
_SIBLING_ROOT="${ROOT}/../llama.cpp"

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

_resolve_sibling_root() {
  echo "${ELIZA_LLAMA_SIBLING:-${_SIBLING_ROOT}}"
}

LLAMA_CPP="$(_resolve_llama_cpp_root)"
SIBLING="$(_resolve_sibling_root)"
CONTAINER_LLAMA="/llama.cpp"
CONTAINER_SIBLING="/llama-sibling"
IMAGE="${CUDA_DEVEL_IMAGE:-docker.io/nvidia/cuda:12.8.0-devel-ubuntu22.04}"
ARCH="${CMAKE_CUDA_ARCHITECTURES:-89-real}"

if [[ ! -f "${LLAMA_CPP}/CMakeLists.txt" ]]; then
  echo "error: llama.cpp missing at ${LLAMA_CPP} — run ./scripts/rebase_vendor_unified.sh --sync" >&2
  exit 1
fi

if [[ ! -f "${SIBLING}/CMakeLists.txt" ]]; then
  if [[ ! -f "${LLAMA_CPP}/ggml/src/ggml-cuda/fused-attn.cu" ]]; then
    echo "error: eliza sibling missing at ${SIBLING} and vendor lacks CUDA fork" >&2
    exit 1
  fi
  echo ">>> vendor has CUDA fork; sibling optional" >&2
  SIBLING="${LLAMA_CPP}"
fi

echo ">>> ensure vendor patches on host (before container build)"
"${ROOT}/scripts/apply_llama_vendor_patches.sh" "${LLAMA_CPP}"

echo ">>> pull ${IMAGE}"
podman pull "${IMAGE}"

echo ">>> build llama-server (source=${LLAMA_CPP}, arch=${ARCH})"
podman run --rm \
  -v "${ROOT}:/zerollama:Z" \
  -v "${LLAMA_CPP}:${CONTAINER_LLAMA}:Z" \
  -v "${SIBLING}:${CONTAINER_SIBLING}:Z" \
  -e LLAMA_CPP_ROOT="${CONTAINER_LLAMA}" \
  -e ELIZA_LLAMA_SIBLING="${CONTAINER_SIBLING}" \
  -e ZEROLLAMA_SKIP_VENDOR_APPLY=1 \
  -e ZEROLLAMA_SKIP_PIN_CHECKOUT=1 \
  -e ZEROLLAMA_SKIP_BUILD_PROBES=1 \
  -e ZEROLLAMA_BUILD_DIR=/llama.cpp/build \
  -e LLAMA_BUILD_UI=OFF \
  -e LLAMA_BUILD_WEBUI=OFF \
  -e LLAMA_USE_PREBUILT_UI=OFF \
  -e CMAKE_CUDA_ARCHITECTURES="${ARCH}" \
  -e LLAMA_BUILD_JOBS=4 \
  "${IMAGE}" \
  bash -lc '
    set -euo pipefail
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq cmake build-essential git rsync curl ca-certificates
    for d in /usr/local/cuda/lib64/stubs /usr/local/cuda/targets/x86_64-linux/lib/stubs; do
      if [[ -f "${d}/libcuda.so" ]]; then
        export LDFLAGS="-L${d} -Wl,-rpath-link,${d} -lcuda ${LDFLAGS:-}"
        echo ">>> CUDA driver stubs: ${d}"
        break
      fi
    done
    rm -rf /tmp/llama-build
    cd /zerollama
    ZEROLLAMA_COMPAT_SRC=/zerollama/llama/compat ZEROLLAMA_COMPAT_COPY=1 \
      ./scripts/stage_llama_compat_for_vendor.sh /llama.cpp
    ./scripts/stage_llama_ext_b8_for_vendor.sh /llama.cpp
    ./scripts/stage_llama_kv_ext_for_vendor.sh /llama.cpp
    set +e
    ./scripts/build_llama_server.sh
    _build_rc=$?
    set -e
    # Build dir is on the mounted vendor tree (ZEROLLAMA_BUILD_DIR=/llama.cpp/build).
    if [[ ! -x /llama.cpp/build/bin/llama-server ]]; then
      echo "error: in-container build did not produce /llama.cpp/build/bin/llama-server" >&2
      exit 1
    fi
    if [[ "${_build_rc}" -ne 0 ]]; then
      echo "warn: in-container build returned ${_build_rc} (host verifies artifacts)" >&2
    fi
  '

BIN="${LLAMA_CPP}/build/bin/llama-server"
if [[ -x "${BIN}" ]]; then
  echo ">>> OK: ${BIN}"
  LD_LIBRARY_PATH="${LLAMA_CPP}/build/bin" "${BIN}" --version 2>&1 | head -1 || true
  readelf -d "${BIN}" 2>/dev/null | grep -E 'RUNPATH|RPATH' || true
  bindir="$(dirname "${BIN}")"
  cuda_lib=""
  for cand in \
    "${bindir}/libggml-cuda.so.0.16.0" \
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
    exit 1
  fi
  _host_cuda_lib_has_symbol() {
    local sym="$1"
    strings "${cuda_lib}" 2>/dev/null | grep -Fq "${sym}" && return 0
    command -v nm >/dev/null 2>&1 || return 1
    nm --demangle "${cuda_lib}" 2>/dev/null | grep -Fq "${sym}" && return 0
    return 1
  }
  if ! _host_cuda_lib_has_symbol 'ggml_cuda_op_set_rows'; then
    echo "error: ${cuda_lib} missing SET_ROWS dispatch" >&2
    exit 1
  fi
  if ! _host_cuda_lib_has_symbol 'fused_attn_qjl_polar_cuda'; then
    echo "error: ${cuda_lib} missing fused QJL attn" >&2
    exit 1
  fi
  echo "OK: libggml-cuda fork KV + fused attn verified"
  if ! strings "${BIN}" | grep -q 'kv/seq-copy'; then
    echo "error: ${BIN} missing /kv/seq-copy route" >&2
    exit 1
  fi
  echo "OK: llama-server embeds /kv/seq-copy"
else
  echo "error: build did not produce ${BIN}" >&2
  exit 1
fi
