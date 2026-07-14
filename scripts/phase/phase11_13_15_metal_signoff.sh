#!/usr/bin/env bash
# Mac sign-off chain: Phase 11 → Phase 13 → Phase 15 (unified memory / Metal).
#
# WHY one script: operators asked for phase order on Mac without CUDA 5080 session;
# each phase has a distinct smoke (admission vs estimate vs native KV).
#
# Self-contained (starts sidecar + Go proxy):
#   METAL_SELF_START=1 M3_LLAMA_MODEL=/path/to/small.gguf ./scripts/phase/phase11_13_15_metal_signoff.sh
#
# Serve already up:
#   ./scripts/phase/phase11_13_15_metal_signoff.sh
#
# Skip Phase 15 GPU steps (CPU KV CI only):
#   PHASE15_LIVE=0 ./scripts/phase/phase11_13_15_metal_signoff.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/runtime/sched_watchdog_env.sh
source "${ROOT}/scripts/runtime/sched_watchdog_env.sh"
export ZEROLLAMA_REPO_ROOT="${ZEROLLAMA_REPO_ROOT:-$ROOT}"
PHASE15_LIVE="${PHASE15_LIVE:-1}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "warn: phase11_13_15_metal_signoff targets Darwin" >&2
fi

if [[ "${METAL_SELF_START:-0}" == "1" ]]; then
  # shellcheck source=scripts/runtime/macos_runtime_serve_lib.sh
  source "${ROOT}/scripts/runtime/macos_runtime_serve_lib.sh"
  # shellcheck source=scripts/runtime/runtime_smoke_lib.sh
  source "${ROOT}/scripts/runtime/runtime_smoke_lib.sh"
  macos_export_llama_cpp_paths
  export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${ROOT}/build/llama-server-darwin/bin/llama-server}"
  if [[ ! -x "${LLAMA_SERVER_BIN}" && -x "${LLAMA_CPP_ROOT}/build/bin/llama-server" ]]; then
    export LLAMA_SERVER_BIN="${LLAMA_CPP_ROOT}/build/bin/llama-server"
  fi
  if [[ ! -f "${LLAMA_CPP_LIB:-}" ]]; then
    echo "METAL_SELF_START=1 needs ${LLAMA_CPP_LIB:-<unset>}; run ./scripts/build/build_llama_server.sh" >&2
    exit 1
  fi
  smoke_m3_resolve_signoff_model
  macos_runtime_start_sidecar "$LLAMA_MODEL" "" 1
  macos_runtime_start_go
  export RUN_E2E_GGUF="${RUN_E2E_GGUF:-$LLAMA_MODEL}"
fi

echo "== Phase 11 (Mac admission) =="
"${ROOT}/scripts/phase/phase11_metal_admission_smoke.sh"

echo ""
echo "== Phase 13 (Mac VRAM estimate) =="
"${ROOT}/scripts/phase/phase13_metal_vram_smoke.sh"

echo ""
echo "== Phase 15 (native KV CI — CPU) =="
"${ROOT}/scripts/phase/phase15_kv_native_ci.sh"

if [[ "${PHASE15_LIVE}" == "1" ]]; then
  echo ""
  echo "== Phase 15 (Metal GPU sign-off) =="
  export M3_LLAMA_MODEL="${M3_LLAMA_MODEL:-${LLAMA_MODEL:-${RUN_E2E_GGUF:-}}}"
  if [[ -z "${M3_LLAMA_MODEL:-}" ]]; then
    echo "skip: Phase 15 live (set M3_LLAMA_MODEL or METAL_SELF_START=1)" >&2
  else
    PHASE15_SKIP_BOOT=1 "${ROOT}/scripts/phase/phase15_metal_signoff.sh"
  fi
  echo ""
  echo "== Phase 15 upstream KV watch =="
  "${ROOT}/scripts/phase/phase15_upstream_kv_watch.sh"
fi

echo ""
echo "PASS: phase11_13_15_metal_signoff"
