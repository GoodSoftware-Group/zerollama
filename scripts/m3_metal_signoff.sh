#!/usr/bin/env bash
# M3 Apple Silicon sign-off: Phase 13 snapshot + Phase 14 inprocess on Metal.
#
# Prerequisite: build Metal llama.cpp once:
#   LLAMA_CPP_ROOT=../llama.cpp ./scripts/build_llama_server.sh
#
# Usage (starts sidecar runtime + Go proxy if not already listening):
#   ./scripts/m3_metal_signoff.sh
#
# Env:
#   M3_LLAMA_MODEL       — GGUF blob path (default: smallest local text GGUF)
#   RUN_E2E_PROXY_MODEL  — pulled tag for render-chat (auto from blob when possible)
#   M3_SKIP_START=1      — assume OLLAMA_HOST / ZEROLLAMA_RUNTIME_URL already up
#   RUN_E2E_PHASE15=1    — also run ./scripts/phase15_metal_signoff.sh (sidecar path)
#   ZEROLLAMA_BIN        — override zerollama binary (repo ./zerollama, then PATH)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/macos_runtime_serve_lib.sh
source "${ROOT}/scripts/macos_runtime_serve_lib.sh"

LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-${ROOT}/../llama.cpp}"
export LLAMA_CPP_LIB="${LLAMA_CPP_LIB:-${LLAMA_CPP_ROOT}/build/bin/libllama.dylib}"
export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${LLAMA_CPP_ROOT}/build/bin/llama-server}"
export GPU_PHASE13_SNAPSHOT_OUT="${GPU_PHASE13_SNAPSHOT_OUT:-/tmp/metal-session.json}"

smoke_m3_resolve_signoff_model
export LLAMA_MODEL_FOR_SERVE="$LLAMA_MODEL"

if [[ ! -f "${LLAMA_CPP_LIB}" ]]; then
  echo "Missing ${LLAMA_CPP_LIB}; run ./scripts/build_llama_server.sh" >&2
  exit 1
fi

export RUN_E2E_PHASE14=1

if [[ "${M3_SKIP_START:-0}" != "1" ]]; then
  trap macos_runtime_sidecar_cleanup EXIT
fi

if [[ "${M3_SKIP_START:-0}" != "1" ]]; then
  macos_runtime_start_sidecar "$LLAMA_MODEL" "" 1
  macos_runtime_start_go
fi

echo "== coordination =="
"${ROOT}/scripts/e2e_coordination_smoke.sh"

echo "== Phase 13 snapshot (metal-unified) =="
export LLAMA_MODEL RUN_E2E_GGUF GPU_PHASE13_SNAPSHOT_OUT
"${ROOT}/scripts/gpu_phase13_snapshot.sh"

echo "== Phase 14 inprocess Metal (apple_silicon.yaml) =="
"${ROOT}/scripts/phase14_yaml_config_smoke.sh"

if [[ "${RUN_E2E_PHASE15:-0}" == "1" ]]; then
  echo ""
  M3_LLAMA_MODEL="$LLAMA_MODEL" PHASE15_SKIP_BOOT=1 "${ROOT}/scripts/phase15_metal_signoff.sh"
fi

echo "PASS: M3 metal sign-off (snapshot: ${GPU_PHASE13_SNAPSHOT_OUT})"
