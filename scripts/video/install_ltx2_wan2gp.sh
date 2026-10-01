#!/usr/bin/env bash
# Install Wan2GP LTX-2.5 weights for zerollama (control-capable A/V T2V).
# Usage:
#   ./scripts/video/install_ltx2_wan2gp.sh                 # distilled int8 + gemma4 + VAEs + union control LoRA
#   ./scripts/video/install_ltx2_wan2gp.sh --weights-only
#   ./scripts/video/install_ltx2_wan2gp.sh --control-only   # union-control LoRA only
#   ./scripts/video/install_ltx2_wan2gp.sh --dry-run
#
# Prefer SSD on astra:
#   WAN2GP_ROOT=/mnt/ssd2/zerollama/third_party/wan2gp
# Never binds Gradio to :11434 / :8081. See docs/ltx-t2v.md.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -z "${WAN2GP_ROOT:-}" ]]; then
  if [[ -d /mnt/ssd2/zerollama/third_party/wan2gp || -d /mnt/ssd2 ]]; then
    WAN2GP_ROOT=/mnt/ssd2/zerollama/third_party/wan2gp
  else
    WAN2GP_ROOT="$HOME/.zerollama/third_party/wan2gp"
  fi
fi
WAN2GP_GIT="${WAN2GP_GIT:-https://github.com/deepbeepmeep/Wan2GP.git}"
HF_REPO="${LTX2_HF_REPO:-DeepBeepMeep/LTX-2}"

sibling_wan2gp() { echo "$(cd "$REPO_ROOT/.." && pwd)/Wan2GP"; }

resolve_wan2gp_repo() {
  if [[ -n "${WAN2GP_REPO:-}" ]]; then
    if [[ -d "$WAN2GP_REPO/.git" || -f "$WAN2GP_REPO/wgp.py" || -d "$WAN2GP_REPO/models/ltx2" ]]; then
      echo "$WAN2GP_REPO"; return
    fi
  fi
  local cand
  for cand in "$(sibling_wan2gp)" /root/Wan2GP "$WAN2GP_ROOT/repo"; do
    if [[ -d "$cand/.git" || -f "$cand/wgp.py" || -d "$cand/models/ltx2" ]]; then
      echo "$cand"; return
    fi
  done
  echo "${WAN2GP_REPO:-$WAN2GP_ROOT/repo}"
}

ensure_wan2gp_repo() {
  if [[ -d "$WAN2GP_REPO/.git" || -f "$WAN2GP_REPO/wgp.py" || -d "$WAN2GP_REPO/models/ltx2" ]]; then
    return 0
  fi
  echo "==> clone Wan2GP -> $WAN2GP_REPO"
  mkdir -p "$(dirname "$WAN2GP_REPO")"
  git clone --depth 1 "$WAN2GP_GIT" "$WAN2GP_REPO"
}

WAN2GP_REPO="$(resolve_wan2gp_repo)"
VENV_DIR="${WAN2GP_VENV:-$WAN2GP_ROOT/venv}"
CKPT_DIR="${WAN2GP_CKPT_DIR:-$WAN2GP_ROOT/ckpts}"
MODE="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --weights-only) MODE=weights; shift ;;
    --control-only) MODE=control; shift ;;
    --venv-only) MODE=venv; shift ;;
    --dry-run) MODE=dryrun; shift ;;
    -h|--help)
      echo "Usage: $0 [--weights-only|--control-only|--venv-only|--dry-run]"
      echo "Env: WAN2GP_REPO WAN2GP_ROOT WAN2GP_VENV WAN2GP_CKPT_DIR LTX2_HF_REPO"
      exit 0
      ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$WAN2GP_ROOT" "$CKPT_DIR/gemma4-12b-ltx-v1" "$CKPT_DIR/loras"
ensure_wan2gp_repo

if [[ ! -d "$WAN2GP_REPO" ]]; then
  echo "Wan2GP repo missing at $WAN2GP_REPO" >&2
  exit 1
fi

if [[ "$(cd "$WAN2GP_REPO" && pwd)" != "$(cd "$WAN2GP_ROOT" 2>/dev/null && pwd)/repo" ]]; then
  ln -sfn "$WAN2GP_REPO" "$WAN2GP_ROOT/repo"
  echo "linked $WAN2GP_ROOT/repo -> $WAN2GP_REPO"
fi
if [[ ! -e "$WAN2GP_REPO/ckpts" ]]; then
  ln -sfn "$CKPT_DIR" "$WAN2GP_REPO/ckpts"
elif [[ -L "$WAN2GP_REPO/ckpts" ]]; then
  ln -sfn "$CKPT_DIR" "$WAN2GP_REPO/ckpts"
fi

# Core distilled + TE + VAEs + connectors + distilled LoRA + union control LoRA.
LTX25_FILES=(
  "ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors"
  "ltx-2.5-22b_video_vae_bf16.safetensors"
  "ltx-2.5-22b_audio_vae_bf16.safetensors"
  "ltx-2.5-22b_vocoder_bf16.safetensors"
  "ltx-2.5-22b_text_embedding_projection_bf16.safetensors"
  "ltx-2.5-22b_video_embeddings_connector_int8_convrot.safetensors"
  "ltx-2.5-22b_audio_embeddings_connector_int8_convrot.safetensors"
  "ltx-2.5-22b_diffusion_video_vae_bf16.safetensors"
  "ltx-2.5-22b-distilled-lora-450_bf16.safetensors"
  "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
  "ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors"
  "gemma4-12b-ltx-v1/gemma4-12b-ltx-v1_int8_convrot.safetensors"
  "gemma4-12b-ltx-v1/config.json"
  "gemma4-12b-ltx-v1/tokenizer.json"
  "gemma4-12b-ltx-v1/tokenizer_config.json"
  "gemma4-12b-ltx-v1/chat_template.jinja"
)

CONTROL_FILES=(
  "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
)

download_files() {
  local mode="$1" # all | control
  python3 - <<PY
from huggingface_hub import hf_hub_download
import os
root = os.path.expanduser("$CKPT_DIR")
os.makedirs(root, exist_ok=True)
os.makedirs(os.path.join(root, "gemma4-12b-ltx-v1"), exist_ok=True)
repo = "$HF_REPO"
all_files = [
    "ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors",
    "ltx-2.5-22b_video_vae_bf16.safetensors",
    "ltx-2.5-22b_audio_vae_bf16.safetensors",
    "ltx-2.5-22b_vocoder_bf16.safetensors",
    "ltx-2.5-22b_text_embedding_projection_bf16.safetensors",
    "ltx-2.5-22b_video_embeddings_connector_int8_convrot.safetensors",
    "ltx-2.5-22b_audio_embeddings_connector_int8_convrot.safetensors",
    "ltx-2.5-22b_diffusion_video_vae_bf16.safetensors",
    "ltx-2.5-22b-distilled-lora-450_bf16.safetensors",
    "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
    "ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors",
    "gemma4-12b-ltx-v1/gemma4-12b-ltx-v1_int8_convrot.safetensors",
    "gemma4-12b-ltx-v1/config.json",
    "gemma4-12b-ltx-v1/tokenizer.json",
    "gemma4-12b-ltx-v1/tokenizer_config.json",
    "gemma4-12b-ltx-v1/chat_template.jinja",
]
control_files = ["ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"]
files = control_files if "$mode" == "control" else all_files
for f in files:
    print("GET", f, flush=True)
    path = hf_hub_download(repo, f, local_dir=root)
    print("OK", path, flush=True)
print("ready under", root)
PY
}

check_weights() {
  local missing=0 f
  for f in "${LTX25_FILES[@]}"; do
    if [[ ! -f "$CKPT_DIR/$f" ]]; then
      echo "MISSING $CKPT_DIR/$f"
      missing=1
    else
      echo "OK $f ($(du -h "$CKPT_DIR/$f" | awk '{print $1}'))"
    fi
  done
  # Config ships in Wan2GP tree.
  if [[ ! -f "$WAN2GP_REPO/models/ltx2/configs/ltx2_25_22b_config.json" ]]; then
    echo "MISSING $WAN2GP_REPO/models/ltx2/configs/ltx2_25_22b_config.json"
    missing=1
  fi
  return "$missing"
}

install_venv() {
  if [[ -x "$VENV_DIR/bin/python3" ]]; then
    echo "venv already present: $VENV_DIR"
    # Re-pin torchaudio to match torch CUDA (Wan2GP import path).
    if [[ "$(uname -s)" != "Darwin" ]]; then
      # shellcheck source=/dev/null
      source "$VENV_DIR/bin/activate"
      local index="${WAN2GP_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
      pip install --force-reinstall --no-deps "torchaudio" --index-url "$index" || true
    fi
    return 0
  fi
  # Delegate to H3/LTX shared installer if present pattern.
  if [[ -x "$REPO_ROOT/scripts/video/install_h3_wan2gp.sh" ]]; then
    WAN2GP_ROOT="$WAN2GP_ROOT" WAN2GP_REPO="$WAN2GP_REPO" WAN2GP_VENV="$VENV_DIR" \
      "$REPO_ROOT/scripts/video/install_h3_wan2gp.sh" --venv-only
    return 0
  fi
  echo "missing venv; run ./scripts/video/install_h3_wan2gp.sh --venv-only first" >&2
  return 1
}

run_dry() {
  check_weights || {
    echo "weights incomplete — run without --dry-run first" >&2
    return 1
  }
  if [[ ! -x "$VENV_DIR/bin/python3" ]]; then
    echo "venv missing — file check only (OK)"
    return 0
  fi
  cd "$REPO_ROOT"
  WAN2GP_REPO="$WAN2GP_REPO" WAN2GP_CKPT_DIR="$CKPT_DIR" \
    LTX_PROMPT='dry-run probe' LTX_SIZE=1280x704 LTX_FRAMES=97 LTX_STEPS=8 \
    LTX_MODEL_TYPE=ltx2_25_22B_distilled \
    LTX_OUTPUT_PATH=/tmp/ltx2-dry.mp4 LTX_DRY_RUN=1 \
    "$VENV_DIR/bin/python3" scripts/video/ltx_video_generate.py
}

case "$MODE" in
  weights) download_files all; check_weights ;;
  control) download_files control ;;
  venv) install_venv ;;
  dryrun) run_dry ;;
  all)
    download_files all
    install_venv
    check_weights
    echo "Next: ./scripts/video/register_ltx_models.sh"
    echo "Dry-run: $0 --dry-run"
    echo "WAN2GP_ROOT=$WAN2GP_ROOT"
    ;;
esac
