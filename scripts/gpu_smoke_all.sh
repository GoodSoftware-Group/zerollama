#!/usr/bin/env bash
# Run coordination + full runtime GPU/proxy smokes (see docs/testing-smoke.md).
#
#   export LLAMA_MODEL LLAMA_SERVER_BIN
#   ./scripts/gpu_smoke_all.sh
#
# Optional:
#   RUN_E2E_GGUF=/path/to/small.q8_0.gguf
#   RUN_E2E_NUM_CTX=4096       # default; lower if VRAM tight on 16GB GPUs
#   RUN_E2E_VRAM_CLAMP=1   # serve must set ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX=auto|1
#   RUN_E2E_TOOLS=1        # /api/chat with tools on :8081 and via :8080 proxy
#   RUN_E2E_PROXY_MODEL=my-local-tag
#   RUN_E2E_LEGACY=1 RUN_E2E_LEGACY_MODEL=llama3.2:3B  # defaults to RUN_E2E_PROXY_MODEL or llama3.2:3B
#   RUN_E2E_PREFLIGHT=1  # run ./scripts/phase12_golden_ci.sh before GPU steps
#   RUN_E2E_PHASE13_SNAPSHOT=1  # write gpu_phase13_snapshot JSON after smokes
#   GPU_PHASE13_SNAPSHOT_OUT=/tmp/phase13-post-smoke.json
#   RUN_E2E_TRAINING_OPS=1     # GET /api/train/* (+ optional RUN_E2E_TRAINING_TCP=1)
#
# Phase 14 in-process backends (separate script; serve must use the backend env):
#   ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess|llama-cpp-python ./scripts/phase14_backend_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime_smoke_lib.sh"

export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:8080}"
export ZEROLLAMA_RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"

if [[ "${RUN_E2E_PREFLIGHT:-0}" == "1" ]]; then
  echo "== Phase 12 preflight (no GPU) =="
  "${ROOT}/scripts/phase12_golden_ci.sh"
fi

echo "== coordination smoke =="
"${ROOT}/scripts/e2e_coordination_smoke.sh"

if [[ "${RUN_E2E_TRAINING_OPS:-0}" == "1" ]]; then
  echo "== training ops smoke (HTTP; no job submit) =="
  train_env=()
  [[ "${RUN_E2E_TRAINING_TCP:-0}" == "1" ]] && train_env+=(RUN_E2E_TRAINING_TCP=1)
  # shellcheck disable=SC2086
  env "${train_env[@]}" "${ROOT}/scripts/e2e_training_ops_smoke.sh"
fi

echo "== runtime resume (clear training handoff if any) =="
runtime_resume_if_needed

echo "== VRAM prep (evict stale ggml runners for runtime load) =="
smoke_prepare_vram_for_runtime

echo "== runtime GPU + proxy smoke =="
if [[ "${RUN_E2E_LEGACY:-0}" == "1" ]]; then
  export RUN_E2E_LEGACY_MODEL="${RUN_E2E_LEGACY_MODEL:-${RUN_E2E_PROXY_MODEL:-llama3.2:3B}}"
fi

e2e_env=(RUN_E2E_GPU=1 RUN_E2E_PROXY=1 RUN_E2E_LEGACY=0 RUN_E2E_LEGACY_ONLY=0)
[[ "${RUN_E2E_TOOLS:-0}" == "1" ]] && e2e_env+=(RUN_E2E_TOOLS=1)
[[ "${RUN_E2E_VRAM_CLAMP:-0}" == "1" ]] && e2e_env+=(RUN_E2E_VRAM_CLAMP=1)
# shellcheck disable=SC2086
env "${e2e_env[@]}" "${ROOT}/scripts/e2e_runtime_smoke.sh"

if [[ "${RUN_E2E_LEGACY:-0}" == "1" ]]; then
  echo "== legacy ggml smoke (after runtime; uses :8080 ggml runner) =="
  runtime_resume_if_needed
  # shellcheck disable=SC2086
  env RUN_E2E_LEGACY=1 RUN_E2E_LEGACY_ONLY=1 "${ROOT}/scripts/e2e_runtime_smoke.sh"
fi

echo "== GPU health report =="
runtime_resume_if_needed
"${ROOT}/scripts/gpu_health_report.sh"

if [[ "${RUN_E2E_PHASE13_SNAPSHOT:-0}" == "1" ]]; then
  snap_out="${GPU_PHASE13_SNAPSHOT_OUT:-/tmp/phase13-post-smoke.json}"
  echo "== Phase 13 snapshot (${snap_out}) =="
  snap_args=()
  if [[ -n "${RUN_E2E_GGUF:-}" ]]; then
    snap_args=(--gguf "${RUN_E2E_GGUF}" --num-ctx "${RUN_E2E_NUM_CTX:-4096}")
  elif [[ -n "${LLAMA_MODEL:-}" ]]; then
    snap_args=(--gguf "${LLAMA_MODEL}" --num-ctx "${RUN_E2E_NUM_CTX:-4096}")
  fi
  GPU_PHASE13_SNAPSHOT_OUT="$snap_out" "${ROOT}/scripts/gpu_phase13_snapshot.sh" "${snap_args[@]}"
fi

echo "PASS: gpu_smoke_all"
