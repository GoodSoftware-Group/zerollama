#!/usr/bin/env bash
# Build llama-server from elizaOS/llama.cpp fork (ROADMAP L2 evaluation).
#
# WHY a sibling tree, not vendor/ yet: L2 gate requires measured tok/s + VRAM win
# on stock pin (b9611) vs fork before replacing Ollama's patched vendor tree.
# This script keeps evaluation isolated under ../eliza-llama.cpp.
#
# Usage:
#   ./scripts/build_eliza_llama_server.sh
#   ELIZA_LLAMA_CPP_REF=96dd1a8 ./scripts/build_eliza_llama_server.sh
#
# Then point runtime at the fork binary:
#   export LLAMA_SERVER_BIN=$PWD/../eliza-llama.cpp/build/bin/llama-server
#   export ZEROLLAMA_LLAMA_FORK=1   # or auto-probe when binary has --ctx-checkpoints
#
# CUDA (5080 Blackwell): CMAKE_CUDA_ARCHITECTURES=120-real ./scripts/build_eliza_llama_server.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZEROLLAMA_PARENT="$(cd "${ROOT}/.." && pwd)"
ELIZA_LLAMA_CPP_ROOT="${ELIZA_LLAMA_CPP_ROOT:-${ZEROLLAMA_PARENT}/eliza-llama.cpp}"
ELIZA_LLAMA_CPP_REF="${ELIZA_LLAMA_CPP_REF:-96dd1a8466c84bdd419faf3866425260623fb6b0}"
ELIZA_LLAMA_CPP_REPO="${ELIZA_LLAMA_CPP_REPO:-https://github.com/elizaOS/llama.cpp.git}"

if [[ ! -d "${ELIZA_LLAMA_CPP_ROOT}/.git" ]]; then
  echo "Cloning ${ELIZA_LLAMA_CPP_REPO} → ${ELIZA_LLAMA_CPP_ROOT}"
  git clone --depth 1 "${ELIZA_LLAMA_CPP_REPO}" "${ELIZA_LLAMA_CPP_ROOT}"
fi

cd "${ELIZA_LLAMA_CPP_ROOT}"
git fetch origin --tags --force 2>/dev/null || true
if git rev-parse --verify "${ELIZA_LLAMA_CPP_REF}^{commit}" >/dev/null 2>&1; then
  git checkout --force "${ELIZA_LLAMA_CPP_REF}"
else
  echo "Fetching ${ELIZA_LLAMA_CPP_REF} from origin"
  git fetch origin "${ELIZA_LLAMA_CPP_REF}" --depth 1 2>/dev/null || \
    git fetch origin "${ELIZA_LLAMA_CPP_REF}" 2>/dev/null || true
  git checkout --force "${ELIZA_LLAMA_CPP_REF}"
fi

echo "elizaOS/llama.cpp @ $(git rev-parse --short HEAD)"

# Reuse zerollama Metal/CUDA cmake wiring (stock flags; fork adds kernels in-tree).
LLAMA_CPP_ROOT="${ELIZA_LLAMA_CPP_ROOT}" "${ROOT}/scripts/build_llama_server.sh"

BIN="${ELIZA_LLAMA_CPP_ROOT}/build/bin/llama-server"
help_text="$("${BIN}" --help 2>&1 || true)"
if echo "${help_text}" | grep -qE 'ctx-checkpoints|qjl1_256'; then
  echo "OK: fork probe — ${BIN} advertises eliza KV types / ctx-checkpoints"
else
  echo "warn: ${BIN} missing fork markers in --help (wrong ref or incomplete fork?)" >&2
fi

echo ""
echo "Next:"
echo "  export LLAMA_SERVER_BIN=${BIN}"
echo "  export LLAMA_CPP_ROOT=${ELIZA_LLAMA_CPP_ROOT}"
echo "  export LLAMA_CPP_LIB=${ELIZA_LLAMA_CPP_ROOT}/build/bin/libllama.dylib  # or .so on Linux"
echo "  export ZEROLLAMA_LLAMA_FORK=1"
echo "  ./scripts/l2_fork_eval.sh"
