#!/usr/bin/env bash
# Stage / refresh Clef scaffolding for the L6 pin ladder.
#
# Default: refresh llama/clef/ from ../ollama-upstream (sources + deferred patch).
# --wire: copy deferred patch into llama/compat/002-clef.patch ONLY when pin ≥ b11232.
#
# WHY --wire is gated: apply-patch.cmake GLOB_RECURSEs *.patch under llama/compat/;
# upstream 002-clef.patch does not apply on b10615 and would break CMake configure.
#
# Usage:
#   ./scripts/phase/stage_clef_for_pin.sh           # refresh staging
#   ./scripts/phase/stage_clef_for_pin.sh --wire    # pin ≥ b11232 only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="stage"
if [[ "${1:-}" == "--wire" ]]; then
  MODE="wire"
elif [[ -n "${1:-}" ]]; then
  echo "usage: $0 [--wire]" >&2
  exit 2
fi

VERSION="$(tr -d '[:space:]' < "${ROOT}/LLAMA_CPP_VERSION" 2>/dev/null || true)"
UPSTREAM_ROOT="${OLLAMA_UPSTREAM:-${ROOT}/../ollama-upstream}"
SRC_CLEF="${UPSTREAM_ROOT}/llama/clef"
SRC_PATCH="${UPSTREAM_ROOT}/llama/compat/002-clef.patch"
DEST="${ROOT}/llama/clef"

pin_num="$(echo "${VERSION}" | sed -n 's/^b\([0-9][0-9]*\).*/\1/p')"
if [[ -z "${pin_num}" ]]; then
  pin_num=0
fi

stage_refresh() {
  if [[ ! -f "${SRC_CLEF}/clef.cpp" || ! -f "${SRC_PATCH}" ]]; then
    echo "WARN: upstream Clef not found at ${UPSTREAM_ROOT}" >&2
    echo "  keeping existing ${DEST} (if any); clone via ./scripts/gpu/clone_upstream_ollama.sh" >&2
    if [[ ! -f "${DEST}/clef.cpp" ]]; then
      echo "FAIL: no staged sources and no upstream to copy" >&2
      exit 1
    fi
    return 0
  fi
  mkdir -p "${DEST}"
  cp -f "${SRC_CLEF}/clef.h" "${DEST}/clef.h"
  cp -f "${SRC_CLEF}/clef.cpp" "${DEST}/clef.cpp"
  cp -f "${SRC_PATCH}" "${DEST}/upstream-002-clef.patch"
  cat > "${DEST}/UPSTREAM_SOURCE.txt" <<EOF
Provenance (refreshed by scripts/phase/stage_clef_for_pin.sh):

  source tree: ${UPSTREAM_ROOT}
  HEAD:        $(git -C "${UPSTREAM_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)
  files:       llama/clef/clef.{h,cpp}
               llama/compat/002-clef.patch → llama/clef/upstream-002-clef.patch

WHY not under llama/compat/*.patch until pin ≥ b11232:
  apply-patch.cmake would auto-apply and fail on b10615 server-context hunks.
EOF
  echo ">>> refreshed ${DEST} from ${UPSTREAM_ROOT}" >&2
}

wire_compat() {
  if [[ "${pin_num}" -lt 11232 ]]; then
    echo "REFUSE: pin ${VERSION:-unknown} < b11232 — cannot wire llama/compat/002-clef.patch" >&2
    echo "  Clef server wire needs b11232+. See docs/llama-cpp-pin-ladder.md" >&2
    echo "  Staged deferred patch remains at llama/clef/upstream-002-clef.patch" >&2
    exit 1
  fi
  if [[ ! -f "${DEST}/upstream-002-clef.patch" ]]; then
    stage_refresh
  fi
  cp -f "${DEST}/upstream-002-clef.patch" "${ROOT}/llama/compat/002-clef.patch"
  echo ">>> wired ${ROOT}/llama/compat/002-clef.patch (pin=${VERSION})" >&2
  echo "  Next: git apply --check against vendor; refresh hunks if needed;" >&2
  echo "  wire llama/server CMake target_sources for clef.cpp (upstream pattern)." >&2
}

stage_refresh
if [[ "${MODE}" == "wire" ]]; then
  wire_compat
fi

echo "PASS: stage_clef_for_pin (${MODE})"
