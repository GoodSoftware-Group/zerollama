#!/usr/bin/env bash
# Shared Phase 15 runtime env for GPU sign-off and Mac sidecar smokes.
#
# WHY centralize env + build here: sign-off scripts must enable the same C pool,
# native decode, and linked-ext build flags. Prefer sibling ../llama.cpp (has ggml.h
# and matches operator layout) over in-repo vendor stub; build with runtime/.venv
# Python so a second system-Python build does not overwrite the ext with wrong arch.
#
#   source ./scripts/phase15_runtime_kv_env.sh
#   phase15_runtime_kv_env_apply
#
# Sets:
#   ZEROLLAMA_RUNTIME_KV_NATIVE=1     — C block pool when ext is built (falls back to Python)
#   ZEROLLAMA_KV_NATIVE_DECODE=1    — C prefill/step when linked decode loop ext exists (default on)
#   ZEROLLAMA_KV_NATIVE_SAMPLE=1    — C sampling when linked
#
# Build: phase15_runtime_kv_ext_build auto-links libllama when present (v25 setup.py default).
#
# Optional build (when LLAMA_CPP_LIB or LLAMA_CPP_ROOT has libllama):
#   PHASE15_BUILD_KV_EXT=1 phase15_runtime_kv_ext_build
# Do not call set -euo pipefail here: this file is sourced and must not alter
# the caller's shell options.

_PHASE15_RT_KV_ROOT="${_PHASE15_RT_KV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

phase15_runtime_kv_env_apply() {
  # ZEROLLAMA_RUNTIME_KV_NATIVE=1 activates the C block pool when _kv_native.so is built.
  # If the ext is missing the sidecar logs one warning and falls back to Python pool —
  # run phase15_runtime_kv_ext_build first (or set PHASE15_BUILD_KV_EXT=1, the default).
  export ZEROLLAMA_RUNTIME_KV_NATIVE="${ZEROLLAMA_RUNTIME_KV_NATIVE:-1}"
  export ZEROLLAMA_KV_NATIVE_DECODE="${ZEROLLAMA_KV_NATIVE_DECODE:-1}"
  export ZEROLLAMA_KV_NATIVE_SAMPLE="${ZEROLLAMA_KV_NATIVE_SAMPLE:-1}"
}

# v45: export auto-batch env before multiseq sidecar restart so GPU smokes see enabled coordinators.
# WHY separate from kv_env_apply: daily serve keeps auto-batch off; sign-off opt-in only.
phase15_runtime_auto_batch_env_apply() {
  if [[ "${RUN_P15_AUTO_BATCH_ALL:-0}" == "1" ]]; then
    export RUN_P15_AUTO_BATCH=1
    export RUN_P15_STREAM_AUTO_BATCH=1
    export ZEROLLAMA_KV_AUTO_BATCH=1
    export ZEROLLAMA_KV_AUTO_BATCH_STREAM=1
    return 0
  fi
  if [[ "${PHASE15_AUTO_BATCH_SIGNOFF:-0}" == "1" ]]; then
    export ZEROLLAMA_KV_AUTO_BATCH=1
    export ZEROLLAMA_KV_AUTO_BATCH_STREAM=1
    return 0
  fi
  if [[ "${RUN_P15_AUTO_BATCH:-0}" == "1" ]]; then
    export ZEROLLAMA_KV_AUTO_BATCH=1
  fi
  if [[ "${RUN_P15_STREAM_AUTO_BATCH:-0}" == "1" ]]; then
    export ZEROLLAMA_KV_AUTO_BATCH_STREAM=1
  fi
}

phase15_runtime_kv_ext_build() {
  local root="${_PHASE15_RT_KV_ROOT}/runtime"
  local llama_root="${LLAMA_CPP_ROOT:-}"
  if [[ -z "${llama_root}" ]]; then
    if [[ -f "${_PHASE15_RT_KV_ROOT}/../llama.cpp/build/bin/libllama.dylib" \
       || -f "${_PHASE15_RT_KV_ROOT}/../llama.cpp/build/bin/libllama.so" ]]; then
      llama_root="${_PHASE15_RT_KV_ROOT}/../llama.cpp"
    else
      llama_root="${_PHASE15_RT_KV_ROOT}/llama/llama.cpp"
    fi
  fi
  export LLAMA_CPP_ROOT="${llama_root}"
  # Prefer runtime/.venv (3.11+) over system python3 (often 3.9 on Mac/Linux).
  if [[ -z "${RUNTIME_UV_PYTHON:-}" ]]; then
    if [[ -x "${root}/.venv/bin/python" ]]; then
      export RUNTIME_UV_PYTHON="${root}/.venv/bin/python"
    elif command -v uv >/dev/null 2>&1; then
      # shellcheck source=scripts/runtime_uv_venv.sh
      source "${_PHASE15_RT_KV_ROOT}/scripts/runtime_uv_venv.sh"
      runtime_uv_venv
    fi
  fi
  local py="${RUNTIME_UV_PYTHON:-python3}"
  if [[ -z "${LLAMA_CPP_LIB:-}" ]]; then
    if [[ -f "${llama_root}/build/bin/libllama.dylib" ]]; then
      export LLAMA_CPP_LIB="${llama_root}/build/bin/libllama.dylib"
    elif [[ -f "${llama_root}/build/bin/libllama.so" ]]; then
      export LLAMA_CPP_LIB="${llama_root}/build/bin/libllama.so"
    fi
  fi
  echo "== Phase 15: build _kv_native (block pool + page bind) LLAMA_CPP_ROOT=${LLAMA_CPP_ROOT} =="
  (cd "${root}" && rm -rf build && LLAMA_CPP_ROOT="${llama_root}" LLAMA_CPP_LIB="${LLAMA_CPP_LIB:-}" "${py}" setup.py build_ext --inplace)
  echo "== Phase 15: verify linked decode loop =="
  (cd "${root}" && PYTHONPATH=. LLAMA_CPP_ROOT="${llama_root}" "${py}" -c "
from runtime.kv.native_decode_loop import decode_loop_status, native_decode_loop_available

st = decode_loop_status()
if st.get('available'):
    assert st.get('batch_decode_in_c'), st
    assert native_decode_loop_available(), st
print('decode_loop_status:', st)
")
}
