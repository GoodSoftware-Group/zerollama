#!/usr/bin/env bash
# Native FP8 (E4M3/E5M2) CUDA binary probe — no GGUF required.
#
# WHY: docs/cuda-lanes.md P1 native FP8 weights. Prove packaged/vendor
# libggml-cuda embeds FP8 type/kernel markers (MMVQ/MMQ + convert).
#
#   ./scripts/fp8_cuda_probe.sh
#   LLAMA_CPP_ROOT=vendor/llama-cpp-86d86ed4 ./scripts/fp8_cuda_probe.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/linux_runtime_serve_lib.sh
source "${ROOT}/scripts/linux_runtime_serve_lib.sh"

if [[ -z "${LLAMA_CPP_ROOT:-}" ]]; then
  if root="$(l1_vendor_llama_cpp_root "${ROOT}" 2>/dev/null)"; then
    export LLAMA_CPP_ROOT="${root}"
  elif [[ -d /usr/local/lib/ollama ]]; then
    export LLAMA_CPP_ROOT=/usr/local/lib/ollama
  fi
fi
export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${LLAMA_CPP_ROOT:+${LLAMA_CPP_ROOT}/build/bin/llama-server}}"
[[ -x "${LLAMA_SERVER_BIN:-}" ]] || LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-/usr/local/lib/ollama/llama-server}"
linux_runtime_export_llama_ld_path

echo "== FP8 CUDA binary probe =="
echo "LLAMA_CPP_ROOT=${LLAMA_CPP_ROOT:-}"
echo "LLAMA_SERVER_BIN=${LLAMA_SERVER_BIN}"

cands=()
if [[ -n "${LLAMA_CPP_ROOT:-}" ]]; then
  cands+=("${LLAMA_CPP_ROOT}/build/bin/libggml-cuda.so" "${LLAMA_CPP_ROOT}/build/bin"/libggml-cuda.so.*)
fi
cands+=(/usr/local/lib/ollama/cuda_v12/libggml-cuda.so /usr/local/lib/ollama/cuda_v12/libggml-cuda.so.*)
cands+=(/usr/local/lib/ollama/libggml-cuda.so)

found=""
for c in "${cands[@]}"; do
  for f in $c; do
    [[ -f "$f" ]] || continue
    found="$f"
    break 2
  done
done
if [[ -z "${found}" ]]; then
  echo "FAIL: no libggml-cuda.so found" >&2
  exit 1
fi
echo "lib=${found}"

_has() {
  local needle="$1"
  grep -aFq "${needle}" "${found}" 2>/dev/null && return 0
  strings "${found}" 2>/dev/null | grep -Fq "${needle}" && return 0
  return 1
}

ok=1
for needle in fp8_e4m3 fp8_e5m2 dequantize_fp8_e4m3 dequantize_fp8_e5m2; do
  if _has "${needle}"; then
    echo "  OK: ${needle}"
  else
    echo "  MISS: ${needle}" >&2
    ok=0
  fi
done
if [[ "${ok}" -ne 1 ]]; then
  echo "FAIL: missing FP8 markers in ${found}" >&2
  exit 1
fi

# Prefer nm for MMQ type ids (mangled templates may not appear in strings).
# WHY `|| true` on nm: pipefail would fail the whole pipeline when nm exits non-zero.
if command -v nm >/dev/null 2>&1; then
  for tid in 51 52; do
    if ( nm --demangle "${found}" 2>/dev/null || true ) | grep -Fq "(ggml_type)${tid}"; then
      echo "  OK: mmq/mmvq (ggml_type)${tid}"
    else
      echo "  warn: no demangled (ggml_type)${tid} (strip/LTO?); string markers still OK"
    fi
  done
fi

if [[ -x "${LLAMA_SERVER_BIN}" ]]; then
  if ! "${LLAMA_SERVER_BIN}" --help >/dev/null 2>&1; then
    echo "warn: llama-server --help failed (LD_LIBRARY_PATH?); markers still OK" >&2
  else
    echo "llama-server --help: ok"
  fi
fi

echo "PASS: fp8_cuda_probe (E4M3+E5M2 markers present)"
echo "Doc: docs/cuda-lanes.md (P1 Native FP8 weights)"
