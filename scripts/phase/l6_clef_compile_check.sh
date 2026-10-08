#!/usr/bin/env bash
# Object-compile llama/clef/clef.cpp against the current vendor pin headers.
#
# WHY: Proves the Clef joint head (not the server wire) builds on b10615.
# Does not link into llama-server or start inference.
#
# Usage:
#   ./scripts/phase/l6_clef_compile_check.sh
#   L6_CLEF_OBJ=/tmp/clef.o ./scripts/phase/l6_clef_compile_check.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLEF_CPP="${ROOT}/llama/clef/clef.cpp"
CLEF_H="${ROOT}/llama/clef/clef.h"
VERSION="$(tr -d '[:space:]' < "${ROOT}/LLAMA_CPP_VERSION" 2>/dev/null || echo b10615)"
VENDOR="${LLAMA_CPP_ROOT:-${ROOT}/vendor/llama-cpp-${VERSION}}"
OUT="${L6_CLEF_OBJ:-${TMPDIR:-/tmp}/zerollama-clef-${VERSION}.o}"
CXX="${CXX:-c++}"

if [[ ! -f "${CLEF_CPP}" || ! -f "${CLEF_H}" ]]; then
  echo "FAIL: missing llama/clef sources — refresh from ../ollama-upstream" >&2
  exit 1
fi
if [[ ! -f "${VENDOR}/include/llama.h" ]]; then
  echo "FAIL: vendor headers missing at ${VENDOR}" >&2
  echo "  run: make -f Makefile.sync clean apply-patches" >&2
  exit 1
fi

GGML_INC="${VENDOR}/ggml/include"
if [[ ! -d "${GGML_INC}" ]]; then
  GGML_INC="${ROOT}/ml/backend/ggml/ggml/include"
fi

echo "== L6 Clef compile check (pin=${VERSION}) =="
echo "vendor: ${VENDOR}"
echo "output: ${OUT}"

"${CXX}" -std=c++17 -c \
  -I"${VENDOR}/include" \
  -I"${GGML_INC}" \
  -I"${VENDOR}/common" \
  -I"${VENDOR}/src" \
  -I"${ROOT}/llama/clef" \
  "${CLEF_CPP}" \
  -o "${OUT}"

echo "OK: object compiled ($(wc -c < "${OUT}" | tr -d ' ') bytes)"
echo "NOTE: server score_fields wire still requires pin ≥ b11232 + upstream-002-clef.patch"
echo "PASS: l6_clef_compile_check"
