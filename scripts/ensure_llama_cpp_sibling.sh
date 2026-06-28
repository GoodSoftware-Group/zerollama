#!/usr/bin/env bash
# Ensure sibling llama.cpp exists for runtime / llama-server (source or run directly).
#
# Unified tree: one ../llama.cpp checkout from elizaOS/llama.cpp @ LLAMA_CPP_COMMIT.
# WHY eliza base: superset of ggml-org (dflash-draft, QJL/Polar/TBQ, checkpoints) —
# operators build a single llama-server instead of stock + eliza-llama.cpp siblings.
#
# In-process ggml still syncs from vendor/llama-cpp-b9781/ + Ollama patches until
# vendor rebase lands (see docs/gpu-profiles-l2.md).
#
# Usage:
#   source ./scripts/ensure_llama_cpp_sibling.sh && ensure_llama_cpp_sibling
#   ./scripts/ensure_llama_cpp_sibling.sh
#
# Env:
#   LLAMA_CPP_ROOT          — default ${ZEROLLAMA_REPO}/../llama.cpp
#   MAC_SETUP_LLAMA_CLONE=0 — do not clone; print instructions and exit 1
#   LLAMA_CPP_REPO          — default https://github.com/elizaOS/llama.cpp.git
set -euo pipefail

_ENSURE_LLAMA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ZEROLLAMA_REPO="${ZEROLLAMA_REPO:-${_ENSURE_LLAMA_ROOT}}"
LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-${ZEROLLAMA_REPO}/../llama.cpp}"
LLAMA_CPP_REPO="${LLAMA_CPP_REPO:-https://github.com/elizaOS/llama.cpp.git}"
_LLAMA_PIN_FILE="${ZEROLLAMA_REPO}/LLAMA_CPP_VERSION"
_LLAMA_COMMIT_FILE="${ZEROLLAMA_REPO}/LLAMA_CPP_COMMIT"

_ensure_llama_cpp_ref() {
  if [[ -f "${_LLAMA_COMMIT_FILE}" ]]; then
    tr -d '[:space:]' < "${_LLAMA_COMMIT_FILE}"
    return 0
  fi
  if [[ -f "${_LLAMA_PIN_FILE}" ]]; then
    tr -d '[:space:]' < "${_LLAMA_PIN_FILE}"
    return 0
  fi
  echo "c84b30200c8d512c00c9d61c96bed078f1c0024d"
}

ensure_llama_cpp_at_pin() {
  local root="${1:-${LLAMA_CPP_ROOT}}"
  local ref
  ref="$(_ensure_llama_cpp_ref)"

  if [[ ! -d "${root}/.git" ]]; then
    echo "error: ${root} is not a git checkout (cannot pin ${ref})" >&2
    return 1
  fi

  local want=""
  if git -C "${root}" rev-parse --verify "${ref}^{commit}" >/dev/null 2>&1; then
    want="$(git -C "${root}" rev-parse "${ref}^{commit}")"
  else
    echo ">>> fetching llama.cpp ref ${ref}..." >&2
    git -C "${root}" fetch origin "${ref}" --depth 1 2>/dev/null \
      || git -C "${root}" fetch origin --tags --force 2>/dev/null \
      || true
    if ! git -C "${root}" rev-parse --verify "${ref}^{commit}" >/dev/null 2>&1; then
      echo "error: ref ${ref} not found in ${root}" >&2
      return 1
    fi
    want="$(git -C "${root}" rev-parse "${ref}^{commit}")"
  fi

  local head
  head="$(git -C "${root}" rev-parse HEAD)"
  if [[ "${head}" == "${want}" ]]; then
    return 0
  fi

  echo ">>> checking out llama.cpp @ ${ref} (${want:0:12}) in ${root}..." >&2
  git -C "${root}" checkout --force "${ref}"
}

ensure_llama_cpp_sibling() {
  export LLAMA_CPP_ROOT
  local ref
  ref="$(_ensure_llama_cpp_ref)"

  if [[ -f "${LLAMA_CPP_ROOT}/CMakeLists.txt" ]]; then
    if [[ -d "${LLAMA_CPP_ROOT}/.git" ]]; then
      ensure_llama_cpp_at_pin "${LLAMA_CPP_ROOT}" || true
    fi
    echo ">>> llama.cpp present at ${LLAMA_CPP_ROOT}" >&2
    return 0
  fi

  if [[ "${MAC_SETUP_LLAMA_CLONE:-1}" != "1" ]]; then
    echo "error: llama.cpp not found at ${LLAMA_CPP_ROOT}" >&2
    echo "  clone: git clone ${LLAMA_CPP_REPO} ${LLAMA_CPP_ROOT}" >&2
    echo "  pin:   git -C ${LLAMA_CPP_ROOT} checkout ${ref}" >&2
    echo "  or:    MAC_SETUP_LLAMA_CLONE=1 ./scripts/mac_setup.sh" >&2
    return 1
  fi

  local parent
  parent="$(dirname "${LLAMA_CPP_ROOT}")"
  mkdir -p "${parent}"

  echo ">>> cloning ${LLAMA_CPP_REPO} (ref ${ref}) → ${LLAMA_CPP_ROOT}..." >&2
  if git clone --depth 1 "${LLAMA_CPP_REPO}" "${LLAMA_CPP_ROOT}" 2>/dev/null; then
    ensure_llama_cpp_at_pin "${LLAMA_CPP_ROOT}"
    echo ">>> llama.cpp ready at ${LLAMA_CPP_ROOT}" >&2
    return 0
  fi

  rm -rf "${LLAMA_CPP_ROOT}"
  git clone "${LLAMA_CPP_REPO}" "${LLAMA_CPP_ROOT}"
  ensure_llama_cpp_at_pin "${LLAMA_CPP_ROOT}"
  if [[ ! -f "${LLAMA_CPP_ROOT}/CMakeLists.txt" ]]; then
    echo "error: clone at ${LLAMA_CPP_ROOT} missing CMakeLists.txt" >&2
    return 1
  fi
  echo ">>> llama.cpp ready at ${LLAMA_CPP_ROOT}" >&2
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  ensure_llama_cpp_sibling
fi
