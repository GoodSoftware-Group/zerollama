#!/usr/bin/env bash
# Phase 16 edge binary E2E — -tags edge artifact serves with compile-time edge defaults.
#
# WHY: phase16_edge_smoke uses default zerollama + --edge; operators deploy build_zerollama_edge
# artifacts that auto-apply edge env without a CLI flag. This smoke proves that path end-to-end.
#
# Usage:
#   ./scripts/phase/phase16_edge_binary_smoke.sh
#   M3_LLAMA_MODEL=/path/to/small.gguf P16_MODEL=your:tag ./scripts/phase/phase16_edge_binary_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/runtime/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime/runtime_smoke_lib.sh"

OUT="${P16_EDGE_BIN:-/tmp/zerollama-edge-smoke}"
P16_OUT="${P16_OUT:-/tmp/phase16-edge-binary-smoke.json}"

if [[ -n "${CUDA_LLAMA_MODEL:-}" ]]; then
  export M3_LLAMA_MODEL="${CUDA_LLAMA_MODEL}"
fi
smoke_m3_resolve_signoff_model

export P17_MODEL="${P16_MODEL:-${P17_MODEL:-${RUN_E2E_PROXY_MODEL:-}}}"
if [[ -z "${P17_MODEL}" ]]; then
  echo "No pulled tag for blob ${LLAMA_MODEL}; pull a model or set P16_MODEL=your-tag:latest" >&2
  exit 1
fi

if [[ -z "${LLAMA_SERVER_BIN:-}" ]]; then
  for candidate in \
    "${ROOT}/build/llama-server-darwin/bin/llama-server" \
    "${ROOT}/../llama.cpp/build/bin/llama-server" \
    "${ROOT}/../ollama-upstream/build/llama-server-darwin/bin/llama-server"; do
    if [[ -x "${candidate}" ]]; then
      export LLAMA_SERVER_BIN="${candidate}"
      break
    fi
  done
fi
if [[ -z "${LLAMA_SERVER_BIN:-}" || ! -x "${LLAMA_SERVER_BIN}" ]]; then
  echo "Set LLAMA_SERVER_BIN or run ./scripts/build/build_ollama_llama_server_darwin.sh (Mac)" >&2
  exit 1
fi

echo "== Phase 16 edge binary build =="
"${ROOT}/scripts/build/build_zerollama_edge.sh" "${OUT}"

echo ""
echo "== Phase 16 edge binary E2E (plain serve, no --edge flag) =="
# WHY: stale sidecar from prior smokes breaks P17_ASSERT_RUNTIME_OFF.
_embed_port="${ZEROLLAMA_RUNTIME_EMBED_PORT:-8081}"
if curl -sf -m 2 "http://127.0.0.1:${_embed_port}/health" >/dev/null 2>&1; then
  echo "stopping stale runtime on :${_embed_port} (edge expects runtime off)"
  lsof -ti ":${_embed_port}" | xargs kill -9 2>/dev/null || true
  sleep 1
fi
P17_HOST="${P17_HOST:-127.0.0.1:11438}" \
P17_BIN="${OUT}" \
P17_SERVE_EXTRA="" \
P17_MODE=edge \
P17_ASSERT_RUNTIME_OFF=1 \
P17_ASSERT_EDGE_BUILD=1 \
P17_OUT="${P16_OUT}" \
M3_LLAMA_MODEL="${LLAMA_MODEL}" \
P17_MODEL="${P17_MODEL}" \
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN}" \
"${ROOT}/scripts/phase/phase17_llama_server_smoke.sh"

echo ""
echo "PASS: phase16_edge_binary_smoke (${OUT})"
echo "Doc: docs/phase16-thin-edge.md"
