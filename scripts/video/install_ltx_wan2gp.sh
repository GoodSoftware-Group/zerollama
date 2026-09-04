#!/usr/bin/env bash
# Install Wan2GP lab env + LTXV weights for zerollama backend=ltx.
# Usage:
#   ./scripts/video/install_ltx_wan2gp.sh
#   ./scripts/video/install_ltx_wan2gp.sh --weights-only
#   ./scripts/video/install_ltx_wan2gp.sh --2b-only
#   ./scripts/video/install_ltx_wan2gp.sh --dry-run
#   ./scripts/video/install_ltx_wan2gp.sh --venv-only
#
# Never binds Gradio to :11434 / :8081. See docs/ltx-t2v.md.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WAN2GP_ROOT="${WAN2GP_ROOT:-$HOME/.zerollama/third_party/wan2gp}"
WAN2GP_GIT="${WAN2GP_GIT:-https://github.com/deepbeepmeep/Wan2GP.git}"

sibling_wan2gp() {
  echo "$(cd "$REPO_ROOT/.." && pwd)/Wan2GP"
}

resolve_wan2gp_repo() {
  if [[ -n "${WAN2GP_REPO:-}" ]]; then
    if [[ -d "$WAN2GP_REPO/.git" || -f "$WAN2GP_REPO/wgp.py" || -d "$WAN2GP_REPO/models/ltx_video" ]]; then
      echo "$WAN2GP_REPO"
      return
    fi
  fi
  local cand
  for cand in "$(sibling_wan2gp)" /root/Wan2GP "$WAN2GP_ROOT/repo"; do
    if [[ -d "$cand/.git" || -f "$cand/wgp.py" || -d "$cand/models/ltx_video" ]]; then
      echo "$cand"
      return
    fi
  done
  # Prefer Mac/Linux sibling path; never default to /root on Darwin.
  if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "$(sibling_wan2gp)"
  else
    echo "${WAN2GP_REPO:-/root/Wan2GP}"
  fi
}

ensure_wan2gp_repo() {
  if [[ -d "$WAN2GP_REPO/.git" || -f "$WAN2GP_REPO/wgp.py" || -d "$WAN2GP_REPO/models/ltx_video" ]]; then
    return 0
  fi
  echo "==> clone Wan2GP -> $WAN2GP_REPO"
  mkdir -p "$(dirname "$WAN2GP_REPO")"
  if git clone --depth 1 "$WAN2GP_GIT" "$WAN2GP_REPO"; then
    return 0
  fi
  local fallback="$WAN2GP_ROOT/src"
  echo "==> sibling clone failed; trying $fallback"
  mkdir -p "$(dirname "$fallback")"
  git clone --depth 1 "$WAN2GP_GIT" "$fallback"
  WAN2GP_REPO="$fallback"
}

WAN2GP_REPO="$(resolve_wan2gp_repo)"
VENV_DIR="${WAN2GP_VENV:-$WAN2GP_ROOT/venv}"
CKPT_DIR="${WAN2GP_CKPT_DIR:-$WAN2GP_ROOT/ckpts}"
MODE="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --weights-only) MODE=weights; shift ;;
    --2b-only) MODE=2b; shift ;;
    --venv-only) MODE=venv; shift ;;
    --dry-run) MODE=dryrun; shift ;;
    -h|--help)
      echo "Usage: $0 [--weights-only|--2b-only|--venv-only|--dry-run]"
      echo "Env: WAN2GP_REPO WAN2GP_ROOT WAN2GP_VENV WAN2GP_CKPT_DIR WAN2GP_TORCH_INDEX"
      exit 0
      ;;
    \#*) break ;; # zsh interactive: # is not a comment unless setopt interactivecomments
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$WAN2GP_ROOT" "$CKPT_DIR/T5_xxl_1.1"
ensure_wan2gp_repo

if [[ ! -d "$WAN2GP_REPO" ]]; then
  echo "Wan2GP repo missing at $WAN2GP_REPO — clone failed; set WAN2GP_REPO" >&2
  exit 1
fi

# Stable path for manifests (wan2gp_repo: ~/.zerollama/third_party/wan2gp/repo).
if [[ "$(cd "$WAN2GP_REPO" && pwd)" != "$(cd "$WAN2GP_ROOT" 2>/dev/null && pwd)/repo" ]]; then
  ln -sfn "$WAN2GP_REPO" "$WAN2GP_ROOT/repo"
  echo "linked $WAN2GP_ROOT/repo -> $WAN2GP_REPO"
fi

install_2b_defs() {
  mkdir -p "$WAN2GP_REPO/finetunes" "$WAN2GP_REPO/models/ltx_video/configs"
  cp -f "$REPO_ROOT/scripts/video/wan2gp_finetunes/ltxv_2b_distilled.json" \
    "$WAN2GP_REPO/finetunes/ltxv_2b_distilled.json"
  cp -f "$REPO_ROOT/scripts/video/ltx_configs/ltxv-2b-0.9.8-distilled-fp8.yaml" \
    "$WAN2GP_REPO/models/ltx_video/configs/ltxv-2b-0.9.8-distilled-fp8.yaml"
  echo "installed Wan2GP finetune ltxv_2b_distilled"
}

# Symlink ckpts into Wan2GP tree so wgp.py locate_file works with default paths.
if [[ ! -e "$WAN2GP_REPO/ckpts" ]]; then
  ln -sfn "$CKPT_DIR" "$WAN2GP_REPO/ckpts"
  echo "linked $WAN2GP_REPO/ckpts -> $CKPT_DIR"
elif [[ -L "$WAN2GP_REPO/ckpts" ]]; then
  ln -sfn "$CKPT_DIR" "$WAN2GP_REPO/ckpts"
fi

download_weights() {
  python3 - <<PY
from huggingface_hub import hf_hub_download
import os
root = os.path.expanduser("$CKPT_DIR")
os.makedirs(root, exist_ok=True)
os.makedirs(os.path.join(root, "T5_xxl_1.1"), exist_ok=True)
repo = "DeepBeepMeep/LTX_Video"
files = [
    "ltxv_0.9.8_13B_distilled_quanto_bf16_int8.safetensors",
    "ltxv_0.9.7_VAE.safetensors",
    "ltxv_0.9.7_spatial_upscaler.safetensors",
    "ltxv_scheduler.json",
    "T5_xxl_1.1/T5_xxl_1.1_enc_quanto_bf16_int8.safetensors",
    "T5_xxl_1.1/added_tokens.json",
    "T5_xxl_1.1/special_tokens_map.json",
    "T5_xxl_1.1/spiece.model",
    "T5_xxl_1.1/tokenizer_config.json",
]
for f in files:
    print("GET", f, flush=True)
    path = hf_hub_download(repo, f, local_dir=root)
    print("OK", path, flush=True)
print("weights ready under", root)
PY
}

download_2b_weights() {
  python3 - <<PY
from huggingface_hub import hf_hub_download
import os
root = os.path.expanduser("$CKPT_DIR")
os.makedirs(root, exist_ok=True)
os.makedirs(os.path.join(root, "T5_xxl_1.1"), exist_ok=True)
files = [
    ("Lightricks/LTX-Video", "ltxv-2b-0.9.8-distilled-fp8.safetensors"),
    ("DeepBeepMeep/LTX_Video", "ltxv_0.9.8_spatial_upscaler.safetensors"),
    ("DeepBeepMeep/LTX_Video", "ltxv_0.9.7_VAE.safetensors"),
    ("DeepBeepMeep/LTX_Video", "ltxv_scheduler.json"),
    ("DeepBeepMeep/LTX_Video", "T5_xxl_1.1/T5_xxl_1.1_enc_quanto_bf16_int8.safetensors"),
    ("DeepBeepMeep/LTX_Video", "T5_xxl_1.1/added_tokens.json"),
    ("DeepBeepMeep/LTX_Video", "T5_xxl_1.1/special_tokens_map.json"),
    ("DeepBeepMeep/LTX_Video", "T5_xxl_1.1/spiece.model"),
    ("DeepBeepMeep/LTX_Video", "T5_xxl_1.1/tokenizer_config.json"),
]
for repo, f in files:
    print("GET", repo, f, flush=True)
    path = hf_hub_download(repo, f, local_dir=root)
    print("OK", path, flush=True)
print("2B weights ready under", root)
PY
}

check_weights() {
  local missing=0
  local need=(
    "ltxv_0.9.8_13B_distilled_quanto_bf16_int8.safetensors"
    "ltxv_0.9.7_VAE.safetensors"
    "ltxv_0.9.7_spatial_upscaler.safetensors"
    "ltxv_scheduler.json"
    "T5_xxl_1.1/T5_xxl_1.1_enc_quanto_bf16_int8.safetensors"
  )
  for f in "${need[@]}"; do
    if [[ ! -f "$CKPT_DIR/$f" ]]; then
      echo "MISSING $CKPT_DIR/$f"
      missing=1
    else
      echo "OK $f ($(du -h "$CKPT_DIR/$f" | awk '{print $1}'))"
    fi
  done
  return "$missing"
}

check_2b_weights() {
  local missing=0
  local need=(
    "ltxv-2b-0.9.8-distilled-fp8.safetensors"
    "ltxv_0.9.7_VAE.safetensors"
    "ltxv_scheduler.json"
    "T5_xxl_1.1/T5_xxl_1.1_enc_quanto_bf16_int8.safetensors"
  )
  for f in "${need[@]}"; do
    if [[ ! -f "$CKPT_DIR/$f" ]]; then
      echo "MISSING $CKPT_DIR/$f"
      missing=1
    else
      echo "OK $f ($(du -h "$CKPT_DIR/$f" | awk '{print $1}'))"
    fi
  done
  if [[ ! -f "$CKPT_DIR/ltxv_0.9.8_spatial_upscaler.safetensors" && ! -f "$CKPT_DIR/ltxv-spatial-upscaler-0.9.8.safetensors" ]]; then
    echo "MISSING spatial upscaler (ltxv_0.9.8_spatial_upscaler.safetensors)"
    missing=1
  fi
  return "$missing"
}

install_venv() {
  # Prefer reusing the existing Wan torch venv (same cu128 stack) when present.
  local wan_venv="${WAN_VENV:-$HOME/.zerollama/third_party/wan/venv}"
  if [[ -x "$wan_venv/bin/python3" && ! -e "$VENV_DIR" ]]; then
    ln -sfn "$wan_venv" "$VENV_DIR"
    echo "linked $VENV_DIR -> $wan_venv"
    return 0
  fi
  if [[ -x "$VENV_DIR/bin/python3" ]]; then
    echo "venv already present: $VENV_DIR"
    return 0
  fi
  local py="${WAN2GP_PYTHON:-python3.11}"
  if ! command -v "$py" >/dev/null 2>&1; then
    py=python3
  fi
  "$py" -m venv "$VENV_DIR"
  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate"
  pip install -U pip wheel packaging
  pip install "setuptools>=70,<82"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    pip install torch torchvision
  else
    local index="${WAN2GP_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
    pip install "torch" "torchvision" --index-url "$index" || pip install torch torchvision
  fi
  if [[ -f "$WAN2GP_REPO/requirements.txt" ]]; then
    grep -viE '^[[:space:]]*(gradio)([[:space:]]|$)' "$WAN2GP_REPO/requirements.txt" \
      | pip install -r /dev/stdin || true
  fi
  pip install mmgp huggingface_hub einops || true
  echo "venv ready: $VENV_DIR"
}

run_dry() {
  local model_type="${LTX_MODEL_TYPE:-ltxv_distilled}"
  if [[ "$model_type" == *2b* ]]; then
    check_2b_weights || {
      echo "2B weights incomplete — run $0 --2b-only first" >&2
      return 1
    }
  else
    check_weights || {
      echo "weights incomplete — run without --dry-run first" >&2
      return 1
    }
  fi
  if [[ ! -x "$VENV_DIR/bin/python3" ]]; then
    echo "venv missing — file check only (OK). Install with --venv-only for generate."
    return 0
  fi
  # Prefer thin wrapper dry-run (no Gradio/rembg). Full wgp.py --dry-run needs the UI dep tree.
  cd "$REPO_ROOT"
  WAN2GP_REPO="$WAN2GP_REPO" WAN2GP_CKPT_DIR="$CKPT_DIR" \
    LTX_PROMPT='dry-run probe' LTX_SIZE=768x512 LTX_FRAMES=17 LTX_STEPS=6 \
    LTX_MODEL_TYPE="$model_type" \
    LTX_OUTPUT_PATH=/tmp/ltx-dry.mp4 LTX_DRY_RUN=1 \
    "$VENV_DIR/bin/python3" scripts/video/ltx_video_generate.py
}

case "$MODE" in
  weights) download_weights; check_weights ;;
  2b) download_2b_weights; install_2b_defs; check_2b_weights ;;
  venv) install_venv ;;
  dryrun) run_dry ;;
  all)
    download_weights
    install_venv
    check_weights
    echo "Next: ./scripts/video/register_ltx_models.sh"
    echo "2B distilled FP8: $0 --2b-only && register tag ltxv-2b-distilled:lab"
    echo "Dry-run: $0 --dry-run  (or LTX_DRY_RUN=1 on the wrapper)"
    ;;
esac
