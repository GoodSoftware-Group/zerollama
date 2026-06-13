#!/usr/bin/env bash
# Ensure sibling llama.cpp exists for runtime inprocess / sign-off (source or run directly).
#
# Why: mac_setup and doctor expect LLAMA_CPP_ROOT (default ../llama.cpp). Fresh clones
# often lack it; auto-clone avoids a silent build_llama_server failure.
#
# Usage:
#   source ./scripts/ensure_llama_cpp_sibling.sh && ensure_llama_cpp_sibling
#   ./scripts/ensure_llama_cpp_sibling.sh
#
# Env:
#   LLAMA_CPP_ROOT          — default ${ZEROLLAMA_REPO}/../llama.cpp
#   MAC_SETUP_LLAMA_CLONE=0 — do not clone; print instructions and exit 1
#   LLAMA_CPP_REPO          — default https://github.com/ggml-org/llama.cpp.git
set -euo pipefail

_ENSURE_LLAMA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ZEROLLAMA_REPO="${ZEROLLAMA_REPO:-${_ENSURE_LLAMA_ROOT}}"
LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-${ZEROLLAMA_REPO}/../llama.cpp}"
LLAMA_CPP_REPO="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
_LLAMA_PIN_FILE="${ZEROLLAMA_REPO}/LLAMA_CPP_VERSION"

_ensure_llama_cpp_pin() {
  if [[ -f "${_LLAMA_PIN_FILE}" ]]; then
    tr -d '[:space:]' < "${_LLAMA_PIN_FILE}"
    return 0
  fi
  echo "b9611"
}

ensure_llama_cpp_sibling() {
  export LLAMA_CPP_ROOT
  if [[ -f "${LLAMA_CPP_ROOT}/CMakeLists.txt" ]]; then
    echo ">>> llama.cpp present at ${LLAMA_CPP_ROOT}" >&2
    return 0
  fi

  if [[ "${MAC_SETUP_LLAMA_CLONE:-1}" != "1" ]]; then
    echo "error: llama.cpp not found at ${LLAMA_CPP_ROOT}" >&2
    echo "  clone: git clone ${LLAMA_CPP_REPO} ${LLAMA_CPP_ROOT}" >&2
    echo "  pin:   git -C ${LLAMA_CPP_ROOT} checkout $(_ensure_llama_cpp_pin)" >&2
    echo "  or:    MAC_SETUP_LLAMA_CLONE=1 ./scripts/mac_setup.sh" >&2
    return 1
  fi

  local pin parent
  pin="$(_ensure_llama_cpp_pin)"
  parent="$(dirname "${LLAMA_CPP_ROOT}")"
  mkdir -p "${parent}"

  echo ">>> cloning llama.cpp (pin ${pin}) to ${LLAMA_CPP_ROOT}..." >&2
  # Try branch/tag shallow clone first; fall back to default branch + fetch pin.
  # Why: ggml-org tag naming varies; mac_setup should not hard-fail on tag shape alone.
  if git clone --depth 1 --branch "${pin}" "${LLAMA_CPP_REPO}" "${LLAMA_CPP_ROOT}" 2>/dev/null; then
    echo ">>> llama.cpp ready at ${LLAMA_CPP_ROOT}" >&2
    return 0
  fi

  rm -rf "${LLAMA_CPP_ROOT}"
  git clone --depth 1 "${LLAMA_CPP_REPO}" "${LLAMA_CPP_ROOT}"
  if git -C "${LLAMA_CPP_ROOT}" fetch --depth 1 origin "refs/tags/${pin}:refs/tags/${pin}" 2>/dev/null \
    || git -C "${LLAMA_CPP_ROOT}" fetch --depth 1 origin "${pin}" 2>/dev/null; then
    git -C "${LLAMA_CPP_ROOT}" checkout "${pin}" 2>/dev/null || true
  fi
  if [[ ! -f "${LLAMA_CPP_ROOT}/CMakeLists.txt" ]]; then
    echo "error: clone at ${LLAMA_CPP_ROOT} missing CMakeLists.txt" >&2
    return 1
  fi
  echo ">>> llama.cpp ready at ${LLAMA_CPP_ROOT}" >&2
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  ensure_llama_cpp_sibling
fi
