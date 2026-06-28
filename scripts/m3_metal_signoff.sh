#!/usr/bin/env bash
# M3 Apple Silicon sign-off: Phase 13 snapshot + Phase 14 inprocess on Metal.
#
# Why this script exists: prove runtime Metal inprocess (apple_silicon.yaml) + optional
# qwen35 Go ggml + Phase 15 KV on one Metal device — the Mac counterpart to gpu_5080_session.
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
#   RUN_E2E_QWEN35=1     — also run ./scripts/qwen35_mac_smoke.sh (needs RUN_E2E_QWEN35_MODEL, e.g. eliza-1-2b:latest)
#                          Why before Phase 15: Phase 15 stops :8081; qwen35 needs handoff/resume.
#   RUN_E2E_L2=1         — also run ./scripts/l2_full_gate.sh (fork eval + compat + bench)
#   RUN_E2E_L3=1         — also run ./scripts/l3_cache_smoke.sh + gate report
#   ZEROLLAMA_BIN        — override zerollama binary (repo ./zerollama, then PATH)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/macos_runtime_serve_lib.sh
source "${ROOT}/scripts/macos_runtime_serve_lib.sh"

macos_export_llama_cpp_paths
export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${ROOT}/build/llama-server-darwin/bin/llama-server}"
if [[ ! -x "${LLAMA_SERVER_BIN}" && -x "${LLAMA_CPP_ROOT}/build/bin/llama-server" ]]; then
  export LLAMA_SERVER_BIN="${LLAMA_CPP_ROOT}/build/bin/llama-server"
fi
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

# qwen35 before Phase 15: phase15 restarts/kills the :8081 sidecar on exit; qwen35 needs
# runtime handoff + resume while the M3-managed stack is still up.
if [[ "${RUN_E2E_QWEN35:-0}" == "1" ]]; then
  echo ""
  "${ROOT}/scripts/qwen35_mac_smoke.sh"
fi

if [[ "${RUN_E2E_L2:-0}" == "1" ]]; then
  echo ""
  M3_LLAMA_MODEL="$LLAMA_MODEL" "${ROOT}/scripts/l2_full_gate.sh"
fi

if [[ "${RUN_E2E_L3:-0}" == "1" ]]; then
  echo ""
  M3_LLAMA_MODEL="$LLAMA_MODEL" "${ROOT}/scripts/l3_cache_smoke.sh"
  "${ROOT}/scripts/l3_gate_report.sh" "${L3_OUT:-/tmp/l3-cache-smoke.json}"
fi

if [[ "${RUN_E2E_PHASE15:-0}" == "1" ]]; then
  echo ""
  M3_LLAMA_MODEL="$LLAMA_MODEL" PHASE15_SKIP_BOOT=1 "${ROOT}/scripts/phase15_metal_signoff.sh"
fi

echo "PASS: M3 metal sign-off (snapshot: ${GPU_PHASE13_SNAPSHOT_OUT})"
