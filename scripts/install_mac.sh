#!/usr/bin/env bash
# Install the built zerollama binary onto PATH (user-local by default).
#
# Usage:
#   ./scripts/install_mac.sh                 # PREFIX=$HOME/.local
#   PREFIX=/usr/local ./scripts/install_mac.sh
#   ./scripts/install_mac.sh /path/to/zerollama
#
# Env:
#   PREFIX          — install root (default: $HOME/.local)
#   ZEROLLAMA_BIN   — source binary (default: repo-root ./zerollama)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${PREFIX:-${HOME}/.local}"
SRC="${1:-${ZEROLLAMA_BIN:-${ROOT}/zerollama}}"
DEST_DIR="${PREFIX}/bin"
DEST="${DEST_DIR}/zerollama"

if [[ ! -x "${SRC}" ]]; then
  echo "error: binary not found at ${SRC}" >&2
  echo "  build first: ./scripts/build/build_zerollama_mac.sh" >&2
  echo "  or:          ./scripts/runtime/dev_bootstrap.sh" >&2
  exit 1
fi

mkdir -p "${DEST_DIR}"
if [[ ! -w "${DEST_DIR}" ]]; then
  echo "error: ${DEST_DIR} is not writable (try PREFIX=\$HOME/.local)" >&2
  exit 1
fi

install -m 0755 "${SRC}" "${DEST}"

# PATH snippet for interactive shells
profile="${HOME}/.zprofile"
marker="# zerollama toolchain (user-local)"
if [[ "${PREFIX}" == "${HOME}/.local" || "${PREFIX}" == "${HOME}/.local/"* ]]; then
  if [[ ! -f "${profile}" ]] || ! grep -qF "${marker}" "${profile}" 2>/dev/null; then
    {
      echo ""
      echo "${marker}"
      echo "export PATH=\"${HOME}/.local/bin:${HOME}/.local/go/bin:\$PATH\""
    } >> "${profile}"
    echo ">>> appended PATH to ${profile}" >&2
  fi
fi

echo ">>> installed ${DEST}" >&2
if command -v zerollama >/dev/null 2>&1 && [[ "$(command -v zerollama)" == "${DEST}" ]]; then
  zerollama --version 2>&1 | head -1 || true
else
  echo ">>> on PATH after: export PATH=\"${DEST_DIR}:\$PATH\"  (or open a new shell)" >&2
  "${DEST}" --version 2>&1 | head -1 || true
fi
