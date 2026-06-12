#!/usr/bin/env bash
# Checkout sibling llama.cpp to the pin in upstream Ollama (LLAMA_CPP_VERSION).
#
# Usage:
#   ./scripts/checkout_llama_cpp_pin.sh
#   LLAMA_CPP_ROOT=~/Sites/inference/llama.cpp ./scripts/checkout_llama_cpp_pin.sh
set -euo pipefail

ZEROLLAMA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="${OLLAMA_UPSTREAM_DIR:-${ZEROLLAMA_ROOT}/../ollama-upstream}"
ROOT="${LLAMA_CPP_ROOT:-${ZEROLLAMA_ROOT}/../llama.cpp}"
PIN_FILE="${UPSTREAM}/LLAMA_CPP_VERSION"

if [[ ! -f "${PIN_FILE}" ]]; then
  echo ">>> missing ${PIN_FILE}; run ./scripts/clone_upstream_ollama.sh first" >&2
  exit 1
fi
if [[ ! -d "${ROOT}/.git" ]]; then
  echo ">>> llama.cpp git repo not found at ${ROOT}" >&2
  exit 1
fi

PIN="$(tr -d '[:space:]' < "${PIN_FILE}")"
echo ">>> checking out llama.cpp pin ${PIN} in ${ROOT}" >&2
git -C "${ROOT}" fetch origin --tags
git -C "${ROOT}" checkout "${PIN}"
git -C "${ROOT}" log -1 --oneline
echo ">>> rebuild: LLAMA_CPP_ROOT=${ROOT} ./scripts/build_llama_server.sh" >&2
