#!/usr/bin/env bash
# Shared env for unified eliza vendor + Ollama compat staging (llama-server builds).
#
# Usage:
#   source ./scripts/vendor/llama_unified_vendor_env.sh
#   llama_unified_vendor_prepare "${ZEROLLAMA_ROOT}"
set -euo pipefail

llama_unified_vendor_prepare() {
  local root="${1:?repo root}"
  local fetch_head
  fetch_head="$(grep '^FETCH_HEAD=' "${root}/Makefile.sync" 2>/dev/null | cut -d= -f2 || echo c84b3020)"
  local vendor="${OLLAMA_LLAMA_CPP_SOURCE:-${root}/vendor/llama-cpp-${fetch_head}}"

  if [[ ! -f "${vendor}/CMakeLists.txt" ]]; then
    echo "error: unified vendor missing at ${vendor}" >&2
    echo "  run: ./scripts/vendor/rebase_vendor_unified.sh --sync" >&2
    return 1
  fi

  export OLLAMA_LLAMA_CPP_SOURCE="${vendor}"
  "${root}/scripts/vendor/ensure_llama_vendor_patches.sh" "${vendor}"
  echo ">>> unified llama.cpp: ${vendor}" >&2
}
