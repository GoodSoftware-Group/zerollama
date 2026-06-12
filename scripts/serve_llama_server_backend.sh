#!/usr/bin/env bash
# Start zerollama with eligible GGUF text routed Go → llama-server (Phase 17).
#
# Prerequisites:
#   ./scripts/build_ollama_llama_server_darwin.sh
#   or reuse ../ollama-upstream/build/llama-server-darwin/bin/llama-server
#
# Usage:
#   ./scripts/serve_llama_server_backend.sh
#   LLAMA_SERVER_BIN=/path/to/llama-server ./scripts/serve_llama_server_backend.sh
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
  "${ROOT}/scripts/build_zerollama_mac.sh" "${BIN}"
fi

if [[ -z "${LLAMA_SERVER_BIN:-}" ]]; then
  for candidate in \
    "${ROOT}/build/llama-server-darwin/bin/llama-server" \
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
  echo ">>> warning: llama-server not found; run ./scripts/build_ollama_llama_server_darwin.sh" >&2
fi

echo ">>> eligible GGUF text → Go → llama-server (upstream shape)" >&2
echo ">>> vision/thinking/embed still use ggml when required" >&2
exec "${BIN}" serve --llama-server-backend "$@"
