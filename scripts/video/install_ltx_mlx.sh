#!/usr/bin/env bash
# Install community ltx-mlx (LTX-Video 0.9.8 2B, Apple Silicon).
# Usage:
#   ./scripts/video/install_ltx_mlx.sh
#   ./scripts/video/install_ltx_mlx.sh --venv-only
#   ./scripts/video/install_ltx_mlx.sh --13b-only
#
# Never binds :11434 / :8081. zsh: do not put # comments on the same command line.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LTX_MLX_ROOT="${LTX_MLX_ROOT:-$HOME/.zerollama/third_party/ltx-mlx}"
LTX_MLX_GIT="${LTX_MLX_GIT:-https://huggingface.co/baisampayans/ltx-mlx}"
MODE="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-only) MODE=venv; shift ;;
    --13b-only) MODE=13b; shift ;;
    --dry-run) MODE=dryrun; shift ;;
    -h|--help)
      echo "Usage: $0 [--venv-only|--weights-only|--13b-only|--dry-run]"
      echo "Env: LTX_MLX_ROOT LTX_MLX_SRC LTX_MLX_GIT"
      exit 0
      ;;
    \#*) break ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

sibling="$(cd "$REPO_ROOT/.." && pwd)/ltx-mlx"
SRC="${LTX_MLX_SRC:-}"
if [[ -z "$SRC" ]]; then
  if [[ -f "$sibling/pyproject.toml" ]]; then
    SRC="$sibling"
  else
    SRC="$LTX_MLX_ROOT/src"
  fi
fi
VENV="${LTX_MLX_VENV:-$LTX_MLX_ROOT/venv}"
MODELS="${LTX_MLX_MODEL_DIR:-$LTX_MLX_ROOT/models/LTX}"

mkdir -p "$LTX_MLX_ROOT" "$MODELS"

ensure_src() {
  if [[ -f "$SRC/pyproject.toml" ]]; then
    echo "ltx-mlx source: $SRC"
    return 0
  fi
  echo "==> clone ltx-mlx -> $SRC"
  mkdir -p "$(dirname "$SRC")"
  git clone --depth 1 "$LTX_MLX_GIT" "$SRC"
}

pick_python() {
  local cand
  for cand in "${LTX_MLX_PYTHON:-}" python3.12 python3.11 python3.10 python3; do
    [[ -z "$cand" ]] && continue
    if command -v "$cand" >/dev/null 2>&1; then
      if "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
        echo "$cand"
        return 0
      fi
    fi
  done
  echo "need Python >= 3.10 (found only 3.9?). Install python3.11+" >&2
  return 1
}

install_venv() {
  local py
  py="$(pick_python)"
  if [[ -x "$VENV/bin/python3" ]]; then
    if "$VENV/bin/python3" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
      echo "venv already present: $VENV"
      # shellcheck source=/dev/null
      source "$VENV/bin/activate"
      pip install -e "$SRC"
      pip install huggingface_hub
      return 0
    fi
    echo "replacing Python 3.9 venv (ltx-mlx needs >=3.10)"
    rm -rf "$VENV"
  fi
  "$py" -m venv "$VENV"
  # shellcheck source=/dev/null
  source "$VENV/bin/activate"
  pip install -U pip wheel
  pip install -e "$SRC"
  pip install huggingface_hub
  echo "venv ready: $VENV ($py)"
}

download_2b() {
  # shellcheck source=/dev/null
  source "$VENV/bin/activate"
  echo "==> download 2B distilled + T5 (~16 GiB) -> $MODELS"
  hf download Lightricks/LTX-Video \
    --include "ltxv-2b-0.9.8-distilled.safetensors" \
    --include "text_encoder/*" \
    --include "tokenizer/*" \
    --local-dir "$MODELS"
}

download_13b() {
  # shellcheck source=/dev/null
  source "$VENV/bin/activate"
  local dest="$MODELS/LTX 13B"
  mkdir -p "$dest"
  echo "==> download 13B distilled (~27 GiB) -> $dest"
  hf download Lightricks/LTX-Video \
    --include "ltxv-13b-0.9.8-distilled.safetensors" \
    --local-dir "$dest"
  if [[ -f "$MODELS/ltxv-13b-0.9.8-distilled.safetensors" && ! -f "$dest/ltxv-13b-0.9.8-distilled.safetensors" ]]; then
    mv "$MODELS/ltxv-13b-0.9.8-distilled.safetensors" "$dest/"
  fi
}

check_weights() {
  local missing=0
  if ! ls "$MODELS"/ltxv-2b-0.9.8-distilled*.safetensors >/dev/null 2>&1; then
    if ! ls "$MODELS"/*.safetensors >/dev/null 2>&1; then
      echo "MISSING $MODELS/ltxv-2b-0.9.8-distilled.safetensors"
      missing=1
    fi
  else
    echo "OK 2B distilled ckpt"
  fi
  if [[ ! -f "$MODELS/tokenizer/spiece.model" ]]; then
    echo "MISSING tokenizer/spiece.model"
    missing=1
  fi
  if [[ ! -f "$MODELS/text_encoder/config.json" ]]; then
    echo "MISSING text_encoder/config.json"
    missing=1
  fi
  if [[ "${CHECK_13B:-0}" == 1 ]]; then
    if [[ ! -f "$MODELS/LTX 13B/ltxv-13b-0.9.8-distilled.safetensors" && ! -f "$MODELS/ltxv-13b-0.9.8-distilled.safetensors" ]]; then
      echo "MISSING LTX 13B/ltxv-13b-0.9.8-distilled.safetensors"
      missing=1
    else
      echo "OK 13B distilled ckpt"
    fi
  fi
  return "$missing"
}

run_dry() {
  check_weights || return 1
  LTX_MLX_MODEL_DIR="$MODELS" LTX_PROMPT='dry-run' LTX_SIZE=768x480 \
    LTX_FRAMES=17 LTX_STEPS=4 LTX_OUTPUT_PATH=/tmp/ltx-mlx-dry.mp4 LTX_DRY_RUN=1 \
    "$VENV/bin/python3" "$REPO_ROOT/scripts/video/ltx_mlx_generate.py"
}

ensure_src
case "$MODE" in
  venv) install_venv ;;
  weights) install_venv; download_2b; check_weights ;;
  13b) install_venv; download_2b; download_13b; CHECK_13B=1 check_weights ;;
  dryrun) run_dry ;;
  all)
    install_venv
    download_2b
    check_weights
    echo "Next: ./scripts/video/register_ltx_models.sh"
    echo "Tags: ltxv-2b-mlx:lab (fast)  ltxv-13b-mlx:lab (anime quality, --13b-only)"
    ;;
esac
