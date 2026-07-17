#!/usr/bin/env bash
# Full Apple Silicon sign-off: M3 (Phase 13–14) + optional qwen35 + Phase 15 Metal KV.
#
# Why one script: operators need a single gate for runtime inprocess (daily Mac path) and
# optional qwen35 Go ollama-engine on the same Metal device — without CUDA 5080 session.
#
# Prerequisite: Metal llama.cpp sibling (inprocess + Phase 15):
#   LLAMA_CPP_ROOT=../llama.cpp ./scripts/build/build_llama_server.sh
#
# Usage:
#   ./scripts/gpu/metal_signoff.sh
#   RUN_E2E_QWEN35=1 RUN_E2E_QWEN35_MODEL=eliza-1-2b:latest ./scripts/gpu/metal_signoff.sh
#   # eliza-1-* = ship qwen35 family; qwen3.6:latest also valid but heavier
#
# Same as: RUN_E2E_PHASE15=1 ./scripts/phase/m3_metal_signoff.sh
# Override sign-off text model: M3_LLAMA_MODEL=/path/to/text.gguf ./scripts/gpu/metal_signoff.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RUN_E2E_PHASE15=1
exec "${ROOT}/scripts/phase/m3_metal_signoff.sh"
