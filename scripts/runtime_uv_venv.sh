#!/usr/bin/env bash
# Ensure runtime/.venv via uv (Python 3.11+). Source or run directly.
#
#   source ./scripts/runtime_uv_venv.sh && runtime_uv_venv
#   ./scripts/runtime_uv_venv.sh          # prints .venv/bin/python
#
# Env:
#   RUNTIME_UV_EXTRAS=serve,dev   — pip extras (default)
#   RUNTIME_UV_SYNC=1           — force reinstall editable package
set -euo pipefail

RUNTIME_UV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_UV_VENV="${RUNTIME_UV_ROOT}/runtime/.venv"
RUNTIME_UV_EXTRAS="${RUNTIME_UV_EXTRAS:-serve,dev}"

runtime_uv_venv() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found; install from https://docs.astral.sh/uv/" >&2
    exit 1
  fi
  if [[ ! -x "${RUNTIME_UV_VENV}/bin/python" ]]; then
    echo "Creating runtime venv at ${RUNTIME_UV_VENV} (uv, Python 3.11+)..." >&2
    uv venv "${RUNTIME_UV_VENV}" --python 3.11
  fi
  local py="${RUNTIME_UV_VENV}/bin/python"
  if [[ "${RUNTIME_UV_SYNC:-0}" == "1" ]] || ! "${py}" -c "import fastapi" 2>/dev/null; then
    uv pip install --python "${py}" -e "${RUNTIME_UV_ROOT}/runtime[${RUNTIME_UV_EXTRAS}]"
  fi
  export RUNTIME_UV_PYTHON="${py}"
  export VIRTUAL_ENV="${RUNTIME_UV_VENV}"
  export PATH="${RUNTIME_UV_VENV}/bin:${PATH}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  runtime_uv_venv
  echo "${RUNTIME_UV_PYTHON}"
fi
