#!/usr/bin/env bash
# Ensure .venv-training via uv for embedded training.py (MPS / CUDA).
#
# WHY this script exists: zerollama embeds system libpython at **link time** (CGO). Training
# packages (torch, peft, …) live in $REPO/.venv-training/lib/pythonX.Y/site-packages and are
# prepended to sys.path — the venv interpreter is not the embedded one. **X.Y must match the
# libpython inside the zerollama binary** or torch/_C fails to import (ABI mismatch).
#
# WHY detect from `ldd zerollama` first: pkg-config python3-embed can differ from the linked
# libpython on the same host (e.g. CT 1564: binary → 3.10, pkg-config → 3.11). Serve scripts
# and this helper prefer the binary.
#
# Legacy repo-root venv-training/ is ignored — only .venv-training is canonical (see
# x/pyembed_common/embedded_pytorch_env.c and docs/gpu-training.md). WHY gitignored:
# duplicate ~7GiB torch trees; operators should not recreate this path.
#
#   source ./scripts/training_uv_venv.sh && training_uv_venv
#   ./scripts/training_uv_venv.sh --verify
#   eval "$(./scripts/training_uv_venv.sh --export)"
#   ./scripts/training_uv_venv.sh --embed-py   # print linked libpython version only
#
# Env:
#   UV_BIN                 — path to uv when several copies are on PATH
#   TRAINING_UV_VENV       — venv directory (default: $REPO/.venv-training)
#   TRAINING_UV_PYTHON_VER — uv --python spec (default: embedded_training_python_ver)
#   TRAINING_UV_SYNC=1     — force reinstall requirements-training.txt
#   ZEROLLAMA_BIN          — zerollama path for ldd-based version detect
set -euo pipefail

TRAINING_UV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_UV_VENV="${TRAINING_UV_VENV:-${TRAINING_UV_ROOT}/.venv-training}"

# Return libpython X.Y linked into zerollama (ldd), else pkg-config python3-embed, else 3.11.
embedded_training_python_ver() {
  local bin="${ZEROLLAMA_BIN:-$(command -v zerollama 2>/dev/null || true)}"
  local ver=""
  if [[ -n "$bin" && -x "$bin" ]]; then
    ver="$(ldd "$bin" 2>/dev/null | sed -n 's/.*libpython\([0-9.]*\)\.so.*/\1/p' | head -1 || true)"
  fi
  if [[ -z "$ver" ]] && command -v pkg-config >/dev/null 2>&1 && pkg-config --exists python3-embed 2>/dev/null; then
    ver="$(pkg-config --modversion python3-embed 2>/dev/null || true)"
  fi
  echo "${ver:-3.11}"
}

_embedded_python_spec() {
  embedded_training_python_ver
}

# Match embedded libpython — default 3.11 only when ldd and pkg-config are unavailable (Mac Xcode).
TRAINING_UV_PYTHON_VER="${TRAINING_UV_PYTHON_VER:-$(_embedded_python_spec)}"

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
    --embed-py)
      embedded_training_python_ver
      ;;
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
