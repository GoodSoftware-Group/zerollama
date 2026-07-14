#!/usr/bin/env bash
# Deprecated wrapper — unified llama.cpp lives at ../llama.cpp (elizaOS base).
#
# WHY: zerollama ships one llama-server binary; L2 gates compare L1 vs fork
# GPU profiles on that binary, not stock vs eliza siblings.
#
# Usage (same as before):
#   ./scripts/build/build_eliza_llama_server.sh
#   ELIZA_LLAMA_CPP_ROOT=../llama.cpp ./scripts/build/build_eliza_llama_server.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ZEROLLAMA_PARENT="$(cd "${ROOT}/.." && pwd)"

# Legacy env names map to unified sibling.
export LLAMA_CPP_ROOT="${ELIZA_LLAMA_CPP_ROOT:-${LLAMA_CPP_ROOT:-${ZEROLLAMA_PARENT}/llama.cpp}}"

echo "note: build_eliza_llama_server.sh → unified ../llama.cpp (elizaOS @ LLAMA_CPP_COMMIT)" >&2
exec "${ROOT}/scripts/build/build_llama_server.sh"
