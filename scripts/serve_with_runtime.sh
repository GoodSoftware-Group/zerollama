#!/usr/bin/env bash
# Start Python inference runtime + zerollama serve (Phase 7).
# Requires: runtime venv with [serve], zerollama on PATH, LLAMA_MODEL for GPU inference.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNTIME="${ROOT}/runtime"

# shellcheck source=scripts/runtime_uv_venv.sh
source "${ROOT}/scripts/runtime_uv_venv.sh"
if command -v uv >/dev/null 2>&1; then
  runtime_uv_venv
  exec "${RUNTIME_UV_PYTHON}" -m runtime up "$@"
fi
if [[ -x "${RUNTIME}/.venv/bin/python" ]]; then
  exec "${RUNTIME}/.venv/bin/python" -m runtime up "$@"
fi
if command -v zerollama-runtime >/dev/null 2>&1; then
  exec zerollama-runtime up "$@"
fi
exec python3 -m runtime up "$@"
