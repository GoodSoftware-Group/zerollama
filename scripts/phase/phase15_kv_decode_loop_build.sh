#!/usr/bin/env bash
# Phase 15 v12–v15: optional libllama-linked native ext build + probe + E2E.
#
# v12: link probe (llama_max_devices).
# v13: decode_loop_prefill / decode_loop_step symbols.
# v14: gil_released in status; optional E2E when LLAMA_MODEL + RUN_E2E_DECODE_LOOP=1.
# v15: sampling_in_c; decode_loop_sample symbol.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}/runtime"

: "${LLAMA_CPP_ROOT:=${ROOT}/llama/llama.cpp}"
if [[ ! -d "${LLAMA_CPP_ROOT}" ]]; then
  LLAMA_CPP_ROOT="${ROOT}/../llama.cpp"
fi

if [[ ! -f "${LLAMA_CPP_ROOT}/build/bin/libllama.dylib" && ! -f "${LLAMA_CPP_ROOT}/build/bin/libllama.so" ]]; then
  echo "SKIP: libllama not built under ${LLAMA_CPP_ROOT}; run build_llama_server.sh first" >&2
  exit 0
fi

echo "== Phase 15 v25: build _kv_native (auto-links libllama when present) =="
rm -rf build
_py="${RUNTIME_UV_PYTHON:-python3}"
LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT}" "${_py}" setup.py build_ext --inplace

echo "== Phase 15 v15: decode_loop_status + symbol probe =="
PYTHONPATH=. "${_py}" <<'PY'
from runtime.kv.native_decode_loop import decode_loop_status, native_decode_loop_available

st = decode_loop_status()
assert st["link"] == "native", st
assert st["available"] is True, st
assert int(st.get("llama_max_devices", 0)) >= 1, st
assert st.get("gil_released") is True, st
assert st.get("sampling_in_c") is True, st
assert native_decode_loop_available() is True
print("decode_loop_status:", st)

from runtime.kv import _kv_native
assert hasattr(_kv_native, "decode_loop_prefill"), "decode_loop_prefill missing"
assert hasattr(_kv_native, "decode_loop_step"), "decode_loop_step missing"
assert hasattr(_kv_native, "decode_loop_sample"), "decode_loop_sample missing"
assert hasattr(_kv_native, "page_bind_tensor_probe"), "page_bind_tensor_probe missing (need fork llama-kv-ext)"
from runtime.kv.tensor_probe import tensor_probe_available
print("tensor_probe_available:", tensor_probe_available())
print("decode_loop_prefill:", _kv_native.decode_loop_prefill)
print("decode_loop_step:   ", _kv_native.decode_loop_step)
print("API shape OK")
PY

if [[ -n "${LLAMA_MODEL:-}" && "${RUN_E2E_DECODE_LOOP:-}" == "1" ]]; then
  echo "== Phase 15 v15: linked E2E parity (LLAMA_MODEL set) =="
  export LLAMA_CPP_ROOT
  PYTHONPATH=. python3 -m pytest tests/test_kv_decode_loop_e2e.py -q
else
  echo "SKIP E2E: set RUN_E2E_DECODE_LOOP=1 and LLAMA_MODEL for parity test"
fi

echo "PASS: phase15_kv_decode_loop_build"
