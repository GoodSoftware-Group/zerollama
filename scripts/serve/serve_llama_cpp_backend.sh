#!/usr/bin/env bash
# Start zerollama with GGUF text routed through pinned llama.cpp (Python runtime).
#
# Prerequisites:
#   LLAMA_CPP_ROOT=../llama.cpp with Metal build (./scripts/build/build_llama_server.sh)
#   runtime/.venv (./scripts/runtime/mac_setup.sh or runtime_uv_venv.sh)
#
# Usage:
#   ./scripts/serve/serve_llama_cpp_backend.sh
#   LLAMA_CPP_ROOT=~/Sites/inference/llama.cpp ./scripts/serve/serve_llama_cpp_backend.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export ZEROLLAMA_REPO="${ZEROLLAMA_REPO:-$ROOT}"
export ZEROLLAMA_LLAMA_CPP_BACKEND=1
export LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-${ROOT}/../llama.cpp}"

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

echo ">>> ZEROLLAMA_LLAMA_CPP_BACKEND=1" >&2
echo ">>> LLAMA_CPP_ROOT=${LLAMA_CPP_ROOT}" >&2
echo ">>> eligible GGUF text → Python runtime + llama.cpp (ggml runner skipped on load)" >&2
echo ">>> vision/thinking/embed still use ggml when required" >&2
exec "${BIN}" serve --llama-cpp-backend "$@"
