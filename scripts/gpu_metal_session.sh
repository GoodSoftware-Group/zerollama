#!/usr/bin/env bash
# One-shot Apple Silicon session: Metal smoke + Phase 13 snapshot (+ optional Phase 14).
#
# WHY: gpu_5080_session.sh is CUDA/NVML-centric; Mac operators need the same repeatable
# gate (Phase 10–13) with metal-unified probe and portable JSON — not ad-hoc e2e flags.
#
# Prerequisite — serve in another terminal:
#   zerollama serve
#
# Minimal (coordination + /health metal fields):
#   ./scripts/gpu_metal_session.sh
#
# With runtime infer + snapshot (small GGUF, Metal llama-server):
#   export LLAMA_MODEL=/path/to/small.gguf LLAMA_SERVER_BIN=$(which llama-server)
#   export RUN_E2E_GGUF="$LLAMA_MODEL"
#   ./scripts/gpu_metal_session.sh
#
# Optional:
#   RUN_E2E_GPU=1                    # e2e_runtime_smoke via macos_metal_smoke
#   GPU_PHASE13_SNAPSHOT_OUT=/tmp/metal-session.json
#   RUN_E2E_PHASE14=1 RUN_E2E_INPROCESS=1   # needs LLAMA_CPP_LIB (Metal libllama.dylib)
#   RUN_E2E_PHASE14=1 RUN_E2E_LLAMA_BACKEND_SOURCE=config  # apple_silicon.yaml inprocess
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ZEROLLAMA_REPO_ROOT="${ZEROLLAMA_REPO_ROOT:-$ROOT}"
export GPU_PHASE13_SNAPSHOT_OUT="${GPU_PHASE13_SNAPSHOT_OUT:-/tmp/metal-session.json}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "warn: gpu_metal_session targets Darwin; continuing for CI/bash -n" >&2
fi

echo "== Metal operator smoke =="
if [[ "${RUN_E2E_GPU:-0}" == "1" ]]; then
  if [[ -z "${LLAMA_MODEL:-}" || -z "${LLAMA_SERVER_BIN:-}" ]]; then
    echo "RUN_E2E_GPU=1 needs LLAMA_MODEL and LLAMA_SERVER_BIN (Metal llama-server)" >&2
    exit 1
  fi
  export RUN_E2E_GGUF="${RUN_E2E_GGUF:-$LLAMA_MODEL}"
fi
"${ROOT}/scripts/macos_metal_smoke.sh"

if [[ -n "${LLAMA_MODEL:-${RUN_E2E_GGUF:-}}" ]]; then
  echo ""
  echo "== Phase 13 snapshot (metal-unified) =="
  export GPU_PHASE13_SNAPSHOT_OUT
  "${ROOT}/scripts/gpu_phase13_snapshot.sh" \
    --gguf "${RUN_E2E_GGUF:-$LLAMA_MODEL}"
fi

if [[ "${RUN_E2E_PHASE14:-0}" == "1" ]]; then
  if [[ -z "${LLAMA_MODEL:-}" ]]; then
    echo "RUN_E2E_PHASE14=1 needs LLAMA_MODEL (same GGUF as serve)" >&2
    exit 1
  fi
  echo ""
  echo "== Phase 14 Metal backend smoke =="
  if [[ "${RUN_E2E_INPROCESS:-0}" == "1" && -z "${RUN_E2E_LLAMA_BACKEND_SOURCE:-}" ]]; then
    if [[ -z "${LLAMA_CPP_LIB:-}" ]]; then
      echo "RUN_E2E_INPROCESS=1 needs LLAMA_CPP_LIB (Metal libllama.dylib)" >&2
      exit 1
    fi
    export RUN_E2E_INPROCESS=1 RUN_E2E_LLAMA_BACKEND_SOURCE=env
    "${ROOT}/scripts/phase14_backend_smoke.sh"
  elif [[ "${RUN_E2E_LLAMA_BACKEND_SOURCE:-}" == "config" ]]; then
    export ZEROLLAMA_RUNTIME_CONFIG="${ZEROLLAMA_RUNTIME_CONFIG:-${ROOT}/runtime/configs/apple_silicon.yaml}"
    "${ROOT}/scripts/phase14_yaml_config_smoke.sh"
  else
    phase14_env=()
    [[ "${RUN_E2E_INPROCESS:-0}" == "1" ]] && phase14_env+=(RUN_E2E_INPROCESS=1)
    [[ -n "${RUN_E2E_LLAMA_BACKEND_SOURCE:-}" ]] && phase14_env+=(RUN_E2E_LLAMA_BACKEND_SOURCE="${RUN_E2E_LLAMA_BACKEND_SOURCE}")
    # shellcheck disable=SC2086
    env "${phase14_env[@]}" "${ROOT}/scripts/phase14_backend_smoke.sh"
  fi
fi

echo "PASS: gpu_metal_session (snapshot: ${GPU_PHASE13_SNAPSHOT_OUT})"
