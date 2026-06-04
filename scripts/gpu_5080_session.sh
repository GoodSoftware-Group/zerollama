#!/usr/bin/env bash
# One-shot 5080-class GPU session: preflight goldens + full smoke + Phase 13 snapshot.
#
# WHY this script: CI proves parsers without a GPU; a 16GB host needs one repeatable gate
# (Phase 10–13) with a JSON artifact + gpu_snapshot hints — not ad-hoc e2e flags.
# Does NOT require gpt-oss harmony on host (needs ~40+ GiB RAM); see docs/gpu-5080-operator-guide.md
#
#   export LLAMA_MODEL LLAMA_SERVER_BIN RUN_E2E_GGUF
#   export RUN_E2E_PROXY_MODEL=llama3.2:3B   # optional tools + proxy manifest
#   ./scripts/gpu_5080_session.sh
#
# Optional:
#   RUN_E2E_TOOLS=1 RUN_E2E_LEGACY=1 RUN_E2E_VRAM_CLAMP=1  # forwarded to gpu_smoke_all
#   RUN_E2E_TRAINING_OPS=1 RUN_E2E_TRAINING_TCP=1           # embedded training surfaces (serve needs OLLAMA_TRAINING=true)
#   RUN_E2E_PHASE14=1 RUN_E2E_INPROCESS=1                   # phase14_inprocess_smoke (serve must use inprocess)
#   RUN_E2E_PHASE14_SIGNOFF=1                               # phase14_5080_signoff (needs LLAMA_CPP_LIB; self-contained restarts)
#   RUN_E2E_PHASE15=1                                       # phase15_inprocess_multiseq_smoke (needs LLAMA_CPP_LIB)
#   RUN_E2E_LLAMA_BACKEND_SOURCE=config                      # with phase14_yaml_config_smoke prerequisites
#   RUN_E2E_LLAMA_CPP_PYTHON_GPU=1                           # wheel GPU (with RUN_E2E_LLAMA_CPP_PYTHON=1)
#   GPU_PHASE13_SNAPSHOT_OUT=/tmp/5080-session.json
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ZEROLLAMA_REPO_ROOT="${ZEROLLAMA_REPO_ROOT:-$ROOT}"
export RUN_E2E_PREFLIGHT=1
export RUN_E2E_PHASE13_SNAPSHOT=1
export GPU_PHASE13_SNAPSHOT_OUT="${GPU_PHASE13_SNAPSHOT_OUT:-/tmp/5080-session.json}"

if [[ -z "${LLAMA_MODEL:-}" && -z "${RUN_E2E_GGUF:-}" ]]; then
  echo "Set LLAMA_MODEL or RUN_E2E_GGUF (small GGUF for 16GB, e.g. 1B Q8)" >&2
  exit 1
fi
if [[ -z "${LLAMA_SERVER_BIN:-}" ]]; then
  echo "Set LLAMA_SERVER_BIN (path to llama-server)" >&2
  exit 1
fi

echo "== Phase 12 preflight + GPU smokes + snapshot =="
# Phase 14/15 smokes run after snapshot in this script; suppress during gpu_smoke_all
# so RUN_E2E_PHASE14*=1 does not execute twice (~15–20 min sign-off).
_saved_phase14_signoff="${RUN_E2E_PHASE14_SIGNOFF:-0}"
_saved_phase15="${RUN_E2E_PHASE15:-0}"
_saved_phase14="${RUN_E2E_PHASE14:-0}"
export RUN_E2E_PHASE14_SIGNOFF=0 RUN_E2E_PHASE15=0 RUN_E2E_PHASE14=0
"${ROOT}/scripts/gpu_smoke_all.sh"
export RUN_E2E_PHASE14_SIGNOFF="${_saved_phase14_signoff}"
export RUN_E2E_PHASE15="${_saved_phase15}"
export RUN_E2E_PHASE14="${_saved_phase14}"

if [[ -f "${GPU_PHASE13_SNAPSHOT_OUT}" ]]; then
  echo ""
  (cd "${ROOT}/runtime" && PYTHONPATH=. python3 -m runtime.gpu_snapshot "${GPU_PHASE13_SNAPSHOT_OUT}") || true
fi

if [[ "${RUN_E2E_PHASE14_SIGNOFF:-0}" == "1" ]]; then
  if [[ -z "${LLAMA_CPP_LIB:-}" ]]; then
    echo "RUN_E2E_PHASE14_SIGNOFF=1 requires LLAMA_CPP_LIB (ctypes libllama.so)" >&2
    exit 1
  fi
  echo ""
  echo "== Phase 14/15 5080 sign-off (self-contained) =="
  "${ROOT}/scripts/phase14_5080_signoff.sh"
elif [[ "${RUN_E2E_PHASE15:-0}" == "1" ]]; then
  if [[ -z "${LLAMA_CPP_LIB:-}" ]]; then
    echo "RUN_E2E_PHASE15=1 requires LLAMA_CPP_LIB (ctypes libllama.so)" >&2
    exit 1
  fi
  echo ""
  echo "== Phase 15 in-process multi-seq smoke =="
  "${ROOT}/scripts/phase15_inprocess_multiseq_smoke.sh"
elif [[ "${RUN_E2E_PHASE14:-0}" == "1" ]]; then
  echo ""
  echo "== Phase 14 backend smoke =="
  if [[ "${RUN_E2E_INPROCESS:-0}" == "1" && -z "${RUN_E2E_LLAMA_BACKEND_SOURCE:-}" ]]; then
    "${ROOT}/scripts/phase14_inprocess_smoke.sh"
  elif [[ "${RUN_E2E_LLAMA_CPP_PYTHON:-0}" == "1" && "${RUN_E2E_LLAMA_CPP_PYTHON_GPU:-0}" != "1" && -z "${RUN_E2E_LLAMA_BACKEND_SOURCE:-}" ]]; then
    "${ROOT}/scripts/phase14_wheel_cpu_smoke.sh"
  else
    phase14_env=()
    [[ "${RUN_E2E_INPROCESS:-0}" == "1" ]] && phase14_env+=(RUN_E2E_INPROCESS=1)
    [[ "${RUN_E2E_LLAMA_CPP_PYTHON:-0}" == "1" ]] && phase14_env+=(RUN_E2E_LLAMA_CPP_PYTHON=1)
    [[ "${RUN_E2E_LLAMA_CPP_PYTHON_GPU:-0}" == "1" ]] && phase14_env+=(RUN_E2E_LLAMA_CPP_PYTHON_GPU=1)
    [[ -n "${RUN_E2E_LLAMA_BACKEND_SOURCE:-}" ]] && phase14_env+=(RUN_E2E_LLAMA_BACKEND_SOURCE="${RUN_E2E_LLAMA_BACKEND_SOURCE}")
    # shellcheck disable=SC2086
    env "${phase14_env[@]}" "${ROOT}/scripts/phase14_backend_smoke.sh"
  fi
fi

echo "PASS: gpu_5080_session (snapshot: ${GPU_PHASE13_SNAPSHOT_OUT})"
