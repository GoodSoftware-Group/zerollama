#!/usr/bin/env bash
# Pin CGO python3-embed to a specific minor version for zerollama build (training + runtime embed).
#
# WHY: Ubuntu `python3-embed` follows the distro default (3.10 on 22.04). Operators who want
# .venv-training on 3.11 (uv default, runtime/.venv parity) must link the binary against
# libpython3.11 — pkg-config alone is not enough without this overlay.
#
# Usage:
#   source ./scripts/training_embed_build_env.sh          # default: 3.11 when python-3.11-embed exists
#   source ./scripts/training_embed_build_env.sh 3.10     # explicit
#   CGO_ENABLED=1 go build -o zerollama .
#
# Env:
#   TRAINING_EMBED_PY — override version (e.g. 3.11)
#   TRAINING_EMBED_PC_DIR — overlay dir (default: $REPO/.cache/pc-embed-overlay)
set -euo pipefail

_TRAINING_EMBED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_TRAINING_EMBED_VER="${1:-${TRAINING_EMBED_PY:-}}"

if [[ -z "${_TRAINING_EMBED_VER}" ]]; then
  if pkg-config --exists python-3.11-embed 2>/dev/null; then
    _TRAINING_EMBED_VER="3.11"
  else
    _TRAINING_EMBED_VER="$(pkg-config --modversion python3-embed 2>/dev/null || echo 3.10)"
  fi
fi

_TRAINING_EMBED_PC="${TRAINING_EMBED_PC_DIR:-${_TRAINING_EMBED_ROOT}/.cache/pc-embed-overlay}"
_PC_SRC=""
for _d in /usr/lib64/pkgconfig /usr/lib/x86_64-linux-gnu/pkgconfig /usr/lib/pkgconfig; do
  if [[ -f "${_d}/python-${_TRAINING_EMBED_VER}-embed.pc" ]]; then
    _PC_SRC="${_d}/python-${_TRAINING_EMBED_VER}-embed.pc"
    break
  fi
done
if [[ -z "${_PC_SRC}" ]]; then
  echo "training_embed_build_env: python-${_TRAINING_EMBED_VER}-embed.pc not found (install python${_TRAINING_EMBED_VER}-dev)" >&2
  return 1 2>/dev/null || exit 1
fi

mkdir -p "${_TRAINING_EMBED_PC}"
cp "${_PC_SRC}" "${_TRAINING_EMBED_PC}/python3-embed.pc"
export PKG_CONFIG_PATH="${_TRAINING_EMBED_PC}${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
export TRAINING_EMBED_PY="${_TRAINING_EMBED_VER}"

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "TRAINING_EMBED_PY=${TRAINING_EMBED_PY}"
  echo "PKG_CONFIG_PATH=${PKG_CONFIG_PATH}"
  echo "python3-embed version: $(pkg-config --modversion python3-embed)"
fi
