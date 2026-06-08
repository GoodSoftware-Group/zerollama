#!/usr/bin/env bash
# Full Apple Silicon sign-off: M3 (Phase 13–14) + Phase 15 Metal KV.
#
# One command for operators after building Metal llama.cpp:
#   LLAMA_CPP_ROOT=../llama.cpp ./scripts/build_llama_server.sh
#   ./scripts/metal_signoff.sh
#
# Same as: RUN_E2E_PHASE15=1 ./scripts/m3_metal_signoff.sh
# Override model: M3_LLAMA_MODEL=/path/to/text.gguf ./scripts/metal_signoff.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RUN_E2E_PHASE15=1
exec "${ROOT}/scripts/m3_metal_signoff.sh"
