#!/usr/bin/env bash
# Phase 15 v19–v20 — tensor bind probe (optional libllama-linked ext).
#
# WHY: v19 scaffolds PA↔llama accounting verify; v20 adds llama-kv-ext cell/tensor
# bind when libllama is rebuilt from the zerollama fork (include/llama-kv-ext.h).
# Prefers runtime/.venv Python — why: system python3 (3.9 universal) overwrites the
# arm64-linked ext built by phase15_runtime_kv_env.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}/runtime"

_py="${RUNTIME_UV_PYTHON:-}"
if [[ -z "${_py}" && -x "${ROOT}/runtime/.venv/bin/python" ]]; then
  _py="${ROOT}/runtime/.venv/bin/python"
fi
_py="${_py:-python3}"

echo "== phase15 tensor bind probe (python=${_py}) =="

if [[ -f "${LLAMA_CPP_LIB:-}" || -f "${LLAMA_CPP_ROOT:-}/build/bin/libllama.dylib" || -f "${LLAMA_CPP_ROOT:-}/build/bin/libllama.so" ]]; then
  # shellcheck source=scripts/phase15_runtime_kv_env.sh
  source "${ROOT}/scripts/phase15_runtime_kv_env.sh"
  phase15_runtime_kv_env_apply
  phase15_runtime_kv_ext_build
else
  ZEROLLAMA_KV_DECODE_LOOP="${ZEROLLAMA_KV_DECODE_LOOP:-0}" "${_py}" setup.py build_ext --inplace >/dev/null
fi

PYTHONPATH=. "${_py}" <<'PY'
from runtime.kv.backend import native_available
from runtime.kv.tensor_probe import export_page_table, tensor_probe_available

assert native_available(), "native ext not built"
from runtime.kv._kv_native import page_bind_clear, page_bind_set

page_bind_clear(0)
page_bind_set(0, 16, [1, 2])
rows = export_page_table(0)
assert len(rows) == 2 and rows[0]["block_id"] == 1
page_bind_clear(0)
print("page_bind_table ok")

if tensor_probe_available():
    print("page_bind_tensor_probe: linked")
else:
    print("page_bind_tensor_probe: skip (build with libllama + ZEROLLAMA_KV_DECODE_LOOP=1)")
PY

echo "PASS: phase15_tensor_bind_probe"
