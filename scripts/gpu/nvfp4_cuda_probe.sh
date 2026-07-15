#!/usr/bin/env bash
# NVFP4 CUDA binary probe — no GGUF required.
#
# WHY: docs/cuda-lanes.md P1 asks for dual-4090 NVFP4 sign-off. Before pulling a
# gpt-oss NVFP4 blob, prove the packaged/vendor CUDA backend embeds NVFP4 type
# markers (generic MMQ path on sm_89; not Blackwell MMA).
#
#   ./scripts/gpu/nvfp4_cuda_probe.sh
#   LLAMA_CPP_ROOT=vendor/llama-cpp-8f114a9b ./scripts/gpu/nvfp4_cuda_probe.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/runtime/linux_runtime_serve_lib.sh
source "${ROOT}/scripts/runtime/linux_runtime_serve_lib.sh"

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

echo "== NVFP4 CUDA binary probe =="
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

# Marker strings from ggml NVFP4 CUDA kernels / type name.
hits="$(strings "${found}" | grep -E 'NVFP4|nvfp4|GGML_TYPE_NVFP4' | sort -u | head -20 || true)"
if [[ -z "${hits}" ]]; then
  echo "FAIL: no NVFP4 markers in ${found}" >&2
  exit 1
fi
echo "markers:"
echo "${hits}" | sed 's/^/  /'

# Optional: llama-server --help should still run (cudart path).
if [[ -x "${LLAMA_SERVER_BIN}" ]]; then
  if ! "${LLAMA_SERVER_BIN}" --help >/dev/null 2>&1; then
    echo "warn: llama-server --help failed (LD_LIBRARY_PATH?); markers still OK" >&2
  else
    echo "llama-server --help: ok"
  fi
fi

echo "PASS: nvfp4_cuda_probe (binary markers present; load an NVFP4 GGUF for lane sign-off)"
echo "Doc: docs/cuda-lanes.md (P1 NVFP4 dual-4090)"
