#!/usr/bin/env bash
# Start zerollama with eligible GGUF text routed Go → llama-server (Phase 17).
#
# Prerequisites:
#   ./scripts/build/build_ollama_llama_server_darwin.sh
#   or reuse ../ollama-upstream/build/llama-server-darwin/bin/llama-server
#
# Usage:
#   ./scripts/serve/serve_llama_server_backend.sh
#   LLAMA_SERVER_BIN=/path/to/llama-server ./scripts/serve/serve_llama_server_backend.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export ZEROLLAMA_REPO="${ZEROLLAMA_REPO:-$ROOT}"

# shellcheck source=scripts/runtime/mac_cgo_env.sh
if [[ "$(uname -s)" == "Darwin" ]]; then
  source "${ROOT}/scripts/runtime/mac_cgo_env.sh"
  mac_cgo_env_warn_path
  mac_cgo_env
fi

BIN="${ROOT}/zerollama"
if [[ ! -x "${BIN}" ]]; then
  echo ">>> building zerollama" >&2
  "${ROOT}/scripts/build/build_zerollama_mac.sh" "${BIN}"
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
  echo ">>> warning: llama-server not found; run ./scripts/build/build_ollama_llama_server_darwin.sh" >&2
fi

echo ">>> eligible GGUF → Go → llama-server when --llama-server-backend (text + vision + thinking)" >&2
echo ">>> Linux serve auto: ZEROLLAMA_LLAMA_SERVER=auto when llama-server on disk" >&2
echo ">>> Phase 16 edge (runtime chat off): ./scripts/serve/serve_edge.sh" >&2
exec "${BIN}" serve --llama-server-backend "$@"
