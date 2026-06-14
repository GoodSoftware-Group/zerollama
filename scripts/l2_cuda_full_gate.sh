#!/usr/bin/env bash
# L2 CUDA full gate orchestrator — probe, compat, bench legs, verdict.
#
# WHY one entrypoint: mirrors l2_full_gate.sh (Darwin/Metal) for RTX 5080-class hosts.
# L2 exit gate requires both Metal and CUDA sign-off before vendor merge.
#
# Usage:
#   CUDA_LLAMA_MODEL=/path/to/small.gguf ./scripts/l2_cuda_full_gate.sh
#   L2_BUILD_FORK=1 L2_RUN_27K=1 L2_RUN_131K_FORK=1 ./scripts/l2_cuda_full_gate.sh
#
# Env:
#   CUDA_LLAMA_MODEL      — GGUF path (required)
#   L2_BUILD_FORK=1       — build ../eliza-llama.cpp first
#   L2_SKIP_BENCH=1       — skip tok/s A/B
#   L2_RUN_27K=1          — also bench at L2_NUM_CTX=26624 (needs ~16G+ model)
#   L2_RUN_131K_FORK=1    — fork-only leg at 131072 (long-ctx VRAM gate)
#   L2_OUT_DIR            — default /tmp/l2-cuda-gate
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${L2_OUT_DIR:-/tmp/l2-cuda-gate}"
mkdir -p "${OUT_DIR}"

export STOCK_LLAMA_CPP_ROOT="${STOCK_LLAMA_CPP_ROOT:-${ROOT}/../llama.cpp}"
export ELIZA_LLAMA_CPP_ROOT="${ELIZA_LLAMA_CPP_ROOT:-${ROOT}/../eliza-llama.cpp}"

if [[ -z "${CUDA_LLAMA_MODEL:-}" && -z "${LLAMA_MODEL:-}" ]]; then
  echo "Set CUDA_LLAMA_MODEL (small GGUF for 16 GB VRAM, e.g. 3B Q8)" >&2
  exit 1
fi
export CUDA_LLAMA_MODEL="${CUDA_LLAMA_MODEL:-${LLAMA_MODEL}}"

echo "== L2 fork eval =="
if [[ "${L2_BUILD_FORK:-0}" == "1" ]]; then
  L2_BUILD_FORK=1 "${ROOT}/scripts/l2_fork_eval.sh"
else
  export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${ELIZA_LLAMA_CPP_ROOT}/build/bin/llama-server}"
  "${ROOT}/scripts/l2_fork_eval.sh"
fi

echo ""
echo "== L2 CUDA runtime compat =="
CUDA_LLAMA_MODEL="${CUDA_LLAMA_MODEL}" "${ROOT}/scripts/l2_cuda_runtime_compat_smoke.sh"

if [[ "${L2_SKIP_BENCH:-0}" != "1" ]]; then
  echo ""
  echo "== L2 CUDA bench (8192 ctx) =="
  L2_CUDA_BENCH_OUT="${OUT_DIR}/bench-8k.json" \
    L2_NUM_CTX=8192 \
    "${ROOT}/scripts/l2_cuda_bench.sh"

  if [[ "${L2_RUN_27K:-0}" == "1" ]]; then
    echo ""
    echo "== L2 CUDA bench (26624 ctx) =="
    L2_CUDA_BENCH_OUT="${OUT_DIR}/bench-27k.json" \
      L2_NUM_CTX=26624 \
      L2_NUM_PREDICT="${L2_NUM_PREDICT:-64}" \
      L2_BENCH_RUNS="${L2_BENCH_RUNS:-1}" \
      "${ROOT}/scripts/l2_cuda_bench.sh"
  fi

  if [[ "${L2_RUN_131K_FORK:-0}" == "1" ]]; then
    echo ""
    echo "== L2 CUDA fork-only long ctx (131072) — timed decode after warmups =="
    # WHY KV blocks: single_gpu.yaml defaults 2048×16=32k tokens; 131k needs 8192 blocks.
    # Use a QJL-compatible GGUF (OuteTTS 1B fails n_embd_head_k); 9B @ 131k needs >16 GiB VRAM.
    export ZEROLLAMA_GPU_PROFILE_CTX=0
    export ZEROLLAMA_KV_NUM_BLOCKS="${ZEROLLAMA_KV_NUM_BLOCKS:-8192}"
    L2_SKIP_STOCK=1 \
      L2_CUDA_BENCH_OUT="${OUT_DIR}/bench-131k-fork.json" \
      L2_NUM_CTX=131072 \
      L2_NUM_PREDICT="${L2_NUM_PREDICT:-64}" \
      L2_BENCH_RUNS="${L2_BENCH_RUNS:-2}" \
      L2_HIGH_CTX_WARMUPS="${L2_HIGH_CTX_WARMUPS:-2}" \
      L2_SKIP_PREFILL=1 \
      "${ROOT}/scripts/l2_cuda_bench.sh"
  fi
fi

echo ""
REPORT_ARGS=()
for f in "${OUT_DIR}"/bench-*.json; do
  [[ -f "$f" ]] && REPORT_ARGS+=("$f")
done
if [[ ${#REPORT_ARGS[@]} -gt 0 ]]; then
  "${ROOT}/scripts/l2_gate_report.sh" "${REPORT_ARGS[@]}" | tee "${OUT_DIR}/verdict.txt"
else
  echo "warn: no bench JSON in ${OUT_DIR}" >&2
fi

echo ""
echo "L2 CUDA artifacts: ${OUT_DIR}/"
echo "Doc: docs/gpu-profiles-l2.md"
