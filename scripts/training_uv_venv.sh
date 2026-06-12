#!/usr/bin/env bash
# Ensure .venv-training via uv (Python 3.11+) for embedded training.py (MPS / CUDA).
#
# The Go binary embeds system libpython3; packages come from PYTHONPATH, not the venv
# interpreter itself. Same pattern as runtime/.venv for the sidecar.
#
#   source ./scripts/training_uv_venv.sh && training_uv_venv
#   ./scripts/training_uv_venv.sh --verify
#   eval "$(./scripts/training_uv_venv.sh --export)"
#
# Env:
#   UV_BIN                 — path to uv when several copies are on PATH
#   TRAINING_UV_VENV       — venv directory (default: $REPO/.venv-training)
#   TRAINING_UV_PYTHON_VER — uv --python spec (default: 3.11)
#   TRAINING_UV_SYNC=1     — force reinstall requirements-training.txt
set -euo pipefail

TRAINING_UV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_UV_VENV="${TRAINING_UV_VENV:-${TRAINING_UV_ROOT}/.venv-training}"
TRAINING_UV_PYTHON_VER="${TRAINING_UV_PYTHON_VER:-3.11}"

_training_uv_bin() {
  if [[ -n "${UV_BIN:-}" ]]; then
    echo "${UV_BIN}"
  elif command -v uv >/dev/null 2>&1; then
    command -v uv
  else
    echo ""
  fi
}

training_uv_venv() {
  local uv_bin
  uv_bin="$(_training_uv_bin)"
  if [[ -z "$uv_bin" ]]; then
    echo "uv not found; set UV_BIN to your uv binary or install from https://docs.astral.sh/uv/" >&2
    exit 1
  fi
  if [[ ! -x "${TRAINING_UV_VENV}/bin/python" ]]; then
    echo "Creating training venv at ${TRAINING_UV_VENV} (uv, Python ${TRAINING_UV_PYTHON_VER})..." >&2
    "${uv_bin}" venv "${TRAINING_UV_VENV}" --python "${TRAINING_UV_PYTHON_VER}"
  fi
  local py="${TRAINING_UV_VENV}/bin/python"
  if [[ "${TRAINING_UV_SYNC:-0}" == "1" ]] || ! "${py}" -c "import torch, peft" 2>/dev/null; then
    if [[ "$(uname -s)" == "Darwin" ]]; then
      "${uv_bin}" pip install --python "${py}" -r "${TRAINING_UV_ROOT}/requirements-training.txt"
    else
      "${uv_bin}" pip install --python "${py}" -r "${TRAINING_UV_ROOT}/requirements-training.txt" \
        --extra-index-url https://download.pytorch.org/whl/cu128
    fi
  fi
  export TRAINING_UV_PYTHON="${py}"
  export VIRTUAL_ENV="${TRAINING_UV_VENV}"
  TRAINING_UV_SITE_PACKAGES="$("${py}" -c 'import site; print(site.getsitepackages()[0])')"
  export TRAINING_UV_SITE_PACKAGES
  export PYTHONPATH="${TRAINING_UV_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
}

training_uv_verify() {
  training_uv_venv
  "${TRAINING_UV_PYTHON}" -c "
import torch, transformers, datasets, peft
dev = 'cuda' if torch.cuda.is_available() else (
    'mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu'
)
print('ok', torch.__version__, 'train_device', dev)
"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  case "${1:-}" in
    --export)
      training_uv_venv
      printf 'export PYTHONPATH=%q\n' "${PYTHONPATH}"
      printf 'export TRAINING_UV_PYTHON=%q\n' "${TRAINING_UV_PYTHON}"
      printf 'export TRAINING_UV_SITE_PACKAGES=%q\n' "${TRAINING_UV_SITE_PACKAGES}"
      ;;
    --verify)
      training_uv_verify
      ;;
    *)
      training_uv_venv
      echo "${TRAINING_UV_PYTHON}"
      ;;
  esac
fi
