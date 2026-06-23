#!/usr/bin/env bash
# Start zerollama with Phase 17 Linux auto routing (plain serve → ZEROLLAMA_LLAMA_SERVER=auto).
#
# Prerequisites (Linux):
#   ./scripts/build_llama_server.sh
#
# Usage:
#   ./scripts/serve_linux_auto.sh
#   LLAMA_SERVER_BIN=/path/to/llama-server ./scripts/serve_linux_auto.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ZEROLLAMA_REPO="${ZEROLLAMA_REPO:-$ROOT}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "serve_linux_auto is Linux-only; on Darwin use ./scripts/serve_llama_server_backend.sh or default ggml serve" >&2
  exit 1
fi

BIN="${ROOT}/zerollama"
if [[ ! -x "${BIN}" ]]; then
  echo ">>> building zerollama" >&2
  (cd "${ROOT}" && go build -o "${BIN}" .)
fi

if [[ -z "${LLAMA_SERVER_BIN:-}" ]]; then
  for candidate in \
    "${ROOT}/build/llama-server/bin/llama-server" \
    "${ROOT}/../llama.cpp/build/bin/llama-server" \
    "${ROOT}/../ollama-upstream/build/llama-server/bin/llama-server"; do
    if [[ -x "${candidate}" ]]; then
      export LLAMA_SERVER_BIN="${candidate}"
      break
    fi
  done
fi

if [[ -n "${LLAMA_SERVER_BIN:-}" ]]; then
  echo ">>> LLAMA_SERVER_BIN=${LLAMA_SERVER_BIN}" >&2
else
  echo ">>> warning: llama-server not found; Linux auto routing stays off until binary is on disk" >&2
fi

echo ">>> Phase 17 Linux auto: plain zerollama serve (ZEROLLAMA_LLAMA_SERVER=auto when discoverable)" >&2
echo ">>> Phase 16 edge (runtime chat off): ./scripts/serve_edge.sh" >&2
exec "${BIN}" serve "$@"
