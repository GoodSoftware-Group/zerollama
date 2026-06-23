#!/usr/bin/env bash
# Start zerollama in Phase 16 edge mode (Go → llama-server, Python runtime chat off).
#
# Prerequisites:
#   ./scripts/build_llama_server.sh   # Linux CUDA
#   ./scripts/build_ollama_llama_server_darwin.sh   # Mac Metal
#
# Usage:
#   ./scripts/serve_edge.sh
#   LLAMA_SERVER_BIN=/path/to/llama-server ./scripts/serve_edge.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ZEROLLAMA_REPO="${ZEROLLAMA_REPO:-$ROOT}"

# shellcheck source=scripts/mac_cgo_env.sh
if [[ "$(uname -s)" == "Darwin" ]]; then
  source "${ROOT}/scripts/mac_cgo_env.sh"
  mac_cgo_env_warn_path
  mac_cgo_env
fi

BIN="${ROOT}/zerollama"
if [[ ! -x "${BIN}" ]]; then
  echo ">>> building zerollama" >&2
  if [[ "$(uname -s)" == "Darwin" ]]; then
    "${ROOT}/scripts/build_zerollama_mac.sh" "${BIN}"
  else
    (cd "${ROOT}" && go build -o "${BIN}" .)
  fi
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

if [[ -n "${LLAMA_SERVER_BIN:-}" ]]; then
  echo ">>> LLAMA_SERVER_BIN=${LLAMA_SERVER_BIN}" >&2
else
  echo ">>> warning: llama-server not found; build sibling tree or set LLAMA_SERVER_BIN" >&2
fi

echo ">>> Phase 16 edge: Go → llama-server for GGUF; runtime chat off; training/Eliza/fleet unchanged" >&2
exec "${BIN}" serve --edge "$@"
