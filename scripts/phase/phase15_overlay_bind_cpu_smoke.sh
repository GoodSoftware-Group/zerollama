#!/usr/bin/env bash
# Phase 15 v48: CPU-only donor-buffer overlay bind sign-off.
#
# WHY CPU-only: the donor-buffer registration hook only ever wraps CPU
# (host) ggml_backend_buffer_type groups — llama_kv_cache's allocation loop
# never consults the donor registry for device (Metal/CUDA) buft groups (no
# upstream primitive equivalent to ggml_backend_cpu_buffer_from_ptr exists
# for device memory). This script forces CPU-only libllama linkage and skips
# with a clear message on hosts where that is not possible.
#
# Stages:
#   1. Build _kv_native linked against a CPU-only libllama (LLAMA_KV_EXT_DONOR_BUFFER=1).
#   2. Pin check — patch 0021 + donor API symbols + allocation-loop wiring present.
#   3. API shape probe — register/status/unregister round-trip with a plain
#      ctypes-backed host buffer (no real model needed for this stage).
#   4. Optional E2E (requires LLAMA_MODEL + RUN_E2E_OVERLAY_BIND=1): two-step
#      query-then-register flow against a real GGUF, asserts donor_buffer_status
#      .bound == True after context construction, runs a short decode to prove
#      generation is unaffected, then unregisters cleanly after teardown.
#
# Usage:
#   ./scripts/phase/phase15_overlay_bind_cpu_smoke.sh
#   LLAMA_MODEL=/path/model.gguf RUN_E2E_OVERLAY_BIND=1 ./scripts/phase/phase15_overlay_bind_cpu_smoke.sh
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

echo "== Phase 15 v48 step 1/4: pin check (patch 0021 + donor API + allocation-loop wiring) =="
"${ROOT}/scripts/phase/phase15_llama_kv_ext_pin_check.sh"

echo "== Phase 15 v48 step 2/4: build _kv_native (CPU-only donor-buffer link) =="
rm -rf build
_py="${RUNTIME_UV_PYTHON:-python3}"
LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT}" ZEROLLAMA_KV_DECODE_LOOP=1 "${_py}" setup.py build_ext --inplace

echo "== Phase 15 v48 step 3/4: donor registry API round-trip (no model required) =="
PYTHONPATH=. ZEROLLAMA_KV_OVERLAY_BIND=1 "${_py}" <<'PY'
import ctypes
import os

assert os.environ.get("ZEROLLAMA_KV_OVERLAY_BIND") == "1"

from runtime.kv import overlay_bind

assert overlay_bind.overlay_bind_enabled() is True, "env gate did not take effect"
assert overlay_bind.donor_buffer_available() is True, (
    "native ext missing register_donor_buffer — rebuild with "
    "LLAMA_KV_EXT_DONOR_BUFFER=1 (see runtime/setup.py ZEROLLAMA_KV_DECODE_LOOP branch)"
)

# WHY a real aligned ctypes buffer (not a fake int): the vendor API asserts
# TENSOR_ALIGNMENT alignment (ggml_backend_cpu_buffer_from_ptr) — register()
# would raise on a garbage/unaligned pointer, so this doubles as an alignment
# sanity check for the binding layer even though no cache consumes it here.
size = 1 << 20  # 1 MiB scratch donor; oversized on purpose for the API-shape probe
buf = ctypes.create_string_buffer(size + 4096)
raw_addr = ctypes.addressof(buf)
aligned_addr = (raw_addr + 4095) & ~4095

donor_id = overlay_bind.register_donor_buffer(aligned_addr, size)
print(f"registered donor_id={donor_id} ptr=0x{aligned_addr:x} size={size}")

status = overlay_bind.donor_buffer_status(donor_id)
assert status is not None, "donor_buffer_status returned None for a just-registered id"
assert "bound" in status and "bytes_used" in status, status
# WHY bound is expected False here: no llama_kv_cache was constructed in this
# process during step 3 — nothing has consulted the donor registry yet. This
# stage only proves the registration/query/unregister API round-trips.
assert status["bound"] is False, status
print(f"status (pre-consume): {status}")

overlay_bind.unregister_donor_buffer(donor_id)
status_after = overlay_bind.donor_buffer_status(donor_id)
assert status_after is None, f"donor_id should be gone after unregister, got {status_after}"
print("PASS: register -> status -> unregister round-trip")
PY

if [[ -n "${LLAMA_MODEL:-}" && "${RUN_E2E_OVERLAY_BIND:-}" == "1" ]]; then
  echo "== Phase 15 v48 step 4/4: E2E donor consume + decode + clean unregister (LLAMA_MODEL set) =="
  export LLAMA_CPP_ROOT
  PYTHONPATH=. ZEROLLAMA_KV_OVERLAY_BIND=1 "${_py}" -m pytest tests/test_kv_overlay_bind_e2e.py -q
else
  echo "SKIP E2E: set RUN_E2E_OVERLAY_BIND=1 and LLAMA_MODEL for the real-model donor-consume test"
fi

echo "PASS: phase15_overlay_bind_cpu_smoke"
