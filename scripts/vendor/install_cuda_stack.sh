#!/usr/bin/env bash
# Universal CUDA stack install: patched vendor llama-server + Python runtime sidecar.
#
# Usage:
#   ./scripts/vendor/install_cuda_stack.sh
#   SKIP_LLAMA_BUILD=1 ./scripts/vendor/install_cuda_stack.sh   # install only (binary already built)
#
# Steps:
#   1. apply_llama_vendor_patches.sh (+ 0020 CUDA fork if sibling present)
#   2. build_llama_server_container.sh (CUDA 12.8 devel; avoids host CUDA skew)
#   3. install_cuda_llama_server.sh → /usr/local/lib/ollama
#   4. deploy_runtime.sh → /opt/zerollama/runtime
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FETCH_HEAD="$(grep '^FETCH_HEAD=' "${ROOT}/Makefile.sync" 2>/dev/null | cut -d= -f2 || echo c84b3020)"
VENDOR="${ROOT}/vendor/llama-cpp-${FETCH_HEAD}"
SIBLING="${ELIZA_LLAMA_SIBLING:-${ROOT}/../llama.cpp}"

echo ">>> [1/4] apply Ollama patches on vendor"
"${ROOT}/scripts/vendor/apply_llama_vendor_patches.sh" "${VENDOR}"

if [[ "${SKIP_LLAMA_BUILD:-0}" != "1" ]] && [[ -f "${SIBLING}/ggml/src/ggml-cuda/fused-attn.cu" ]]; then
  echo ">>> [1b] refresh CUDA fork patch 0020 from sibling"
  "${ROOT}/scripts/vendor/create_cuda_fork_patch.sh"
elif [[ ! -f "${SIBLING}/ggml/src/ggml-cuda/fused-attn.cu" ]]; then
  echo ">>> warn: sibling missing CUDA fork sources — vendor must already include 0020" >&2
  if ! grep -q 'GGML_TYPE_QJL1_256' "${VENDOR}/ggml/src/ggml-cuda/set-rows.cu" 2>/dev/null; then
    echo "error: vendor missing CUDA fork SET_ROWS wiring" >&2
    exit 1
  fi
fi

if [[ "${SKIP_LLAMA_BUILD:-0}" != "1" ]]; then
  echo ">>> [2/4] container build llama-server (vendor tree)"
  LLAMA_CPP_ROOT="${VENDOR}" ELIZA_LLAMA_SIBLING="${SIBLING}" \
    "${ROOT}/scripts/vendor/build_llama_server_container.sh"
else
  echo ">>> [2/4] skip build (SKIP_LLAMA_BUILD=1)"
fi

BIN="${VENDOR}/build/bin/llama-server"
if [[ ! -x "${BIN}" ]]; then
  echo "error: missing ${BIN}" >&2
  exit 1
fi

echo ">>> verify patched binary"
_seq_ok=0
if grep -Fq 'kv/seq-copy' < <(strings "${BIN}"); then
  _seq_ok=1
else
  for _impl in "$(dirname "${BIN}")"/libllama-server-impl*; do
    if [[ -f "${_impl}" ]] && grep -Fq 'kv/seq-copy' < <(strings "${_impl}"); then
      _seq_ok=1
      break
    fi
  done
fi
if [[ "${_seq_ok}" -ne 1 ]]; then
  echo "error: ${BIN} missing /kv/seq-copy (patch 0017; check libllama-server-impl*)" >&2
  exit 1
fi
CUDA_LIB="${VENDOR}/build/bin/libggml-cuda.so.0.12.0"
[[ -f "${CUDA_LIB}" ]] || CUDA_LIB="${VENDOR}/build/bin/libggml-cuda.so.0"
_verify_cuda_fork_symbol() {
  local sym="$1"
  grep -Fq "${sym}" < <(strings "${CUDA_LIB}" 2>/dev/null) && return 0
  command -v nm >/dev/null 2>&1 || return 1
  nm --demangle "${CUDA_LIB}" 2>/dev/null | grep -Fq "${sym}" && return 0
  return 1
}
_verify_cuda_fork_symbol 'ggml_cuda_op_set_rows' || {
  echo "error: ${CUDA_LIB} missing SET_ROWS dispatch" >&2
  exit 1
}
_verify_cuda_fork_symbol 'fused_attn_qjl_polar_cuda' || {
  echo "error: ${CUDA_LIB} missing fused QJL attn" >&2
  exit 1
}

echo ">>> [3/4] install llama-server → /usr/local/lib/ollama"
_run_as_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}
_run_as_root env LLAMA_CPP_ROOT="${VENDOR}" "${ROOT}/scripts/vendor/install_cuda_llama_server.sh"

echo ">>> [4/4] deploy Python runtime"
_run_as_root "${ROOT}/scripts/runtime/deploy_runtime.sh"

echo ">>> verify /ready"
curl -sf -m 20 http://127.0.0.1:8081/ready | head -c 200 || {
  echo "warn: runtime /ready not up — check: systemctl status zerollama-runtime" >&2
}

echo ">>> install_cuda_stack OK"
