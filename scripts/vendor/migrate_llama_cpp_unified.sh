#!/usr/bin/env bash
# Migrate from legacy llama.cpp siblings to unified ../llama.cpp @ LLAMA_CPP_COMMIT.
#
# WHY: eliza-llama.cpp / stock-only siblings caused argv and pin drift (MTP failures).
# This script is read-only except optional clone/build when MIGRATE_BUILD=1.
#
# Usage:
#   ./scripts/vendor/migrate_llama_cpp_unified.sh
#   MIGRATE_BUILD=1 ./scripts/vendor/migrate_llama_cpp_unified.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/vendor/ensure_llama_cpp_sibling.sh
source "${ROOT}/scripts/vendor/ensure_llama_cpp_sibling.sh"

UNIFIED="${LLAMA_CPP_ROOT:-}"
if [[ -z "${UNIFIED}" ]]; then
  _PIN_SHORT="$(tr -d '[:space:]' < "${ROOT}/LLAMA_CPP_VERSION" 2>/dev/null || true)"
  if [[ -n "${_PIN_SHORT}" && -f "${ROOT}/vendor/llama-cpp-${_PIN_SHORT}/CMakeLists.txt" ]]; then
    UNIFIED="${ROOT}/vendor/llama-cpp-${_PIN_SHORT}"
  else
    UNIFIED="${ROOT}/../llama.cpp"
  fi
fi
PIN="$(cat "${ROOT}/LLAMA_CPP_COMMIT" 2>/dev/null | tr -d '[:space:]' || true)"
PIN="${PIN:-c84b30200c8d512c00c9d61c96bed078f1c0024d}"

legacy_path() {
  local var="$1"
  local val="${!var:-}"
  [[ -z "${val}" ]] && return 1
  case "${val}" in
    *eliza-llama.cpp*|*eliza_llama.cpp*|*stock-llama.cpp*|*ollama-upstream*)
      echo "${var}=${val}"
      return 0
      ;;
  esac
  return 1
}

echo "== llama.cpp unification migrate =="
echo "unified tree: ${UNIFIED} @ ${PIN:0:12}"
echo ""

WARN=0
while read -r line; do
  echo "WARN legacy env: ${line}"
  WARN=1
done < <(
  legacy_path LLAMA_CPP_ROOT || true
  legacy_path LLAMA_SERVER_BIN || true
)

if [[ "${WARN}" -eq 0 ]]; then
  echo "OK: no legacy LLAMA_* env detected in this shell"
else
  echo ""
  echo "Fix in your shell profile (.zshrc / .bashrc):"
  echo "  unset LLAMA_SERVER_BIN   # let zerollama discover unified build"
  echo "  export LLAMA_CPP_ROOT=${UNIFIED}"
fi

if [[ ! -f "${UNIFIED}/CMakeLists.txt" ]]; then
  echo ""
  echo "Unified checkout missing. Run:"
  echo "  ./scripts/vendor/ensure_llama_cpp_sibling.sh"
  exit 1
fi

BIN="${UNIFIED}/build/bin/llama-server"
if [[ ! -x "${BIN}" ]]; then
  echo ""
  echo "llama-server not built at ${BIN}"
  if [[ "${MIGRATE_BUILD:-0}" == "1" ]]; then
    echo ">>> MIGRATE_BUILD=1: building..."
    LLAMA_CPP_ROOT="${UNIFIED}" "${ROOT}/scripts/build/build_llama_server.sh"
  else
    echo "  MIGRATE_BUILD=1 $0   # clone pin + build"
    exit 1
  fi
fi

echo ""
echo "Verify:"
echo "  export LLAMA_CPP_ROOT=${UNIFIED}"
echo "  unset LLAMA_SERVER_BIN"
echo "  ./zerollama doctor | rg 'llama.cpp unified|llama-server'"
echo ""
echo "Doc: docs/llama-cpp-unification.md"
