#!/usr/bin/env bash
# Install Wan2GP lab env + MiniMax-H3 weights for zerollama backend=h3 runner=h3-wan2gp.
# Usage:
#   ./scripts/video/install_h3_wan2gp.sh
#   ./scripts/video/install_h3_wan2gp.sh --pruned-only
#   ./scripts/video/install_h3_wan2gp.sh --weights-only
#   ./scripts/video/install_h3_wan2gp.sh --venv-only
#   ./scripts/video/install_h3_wan2gp.sh --dry-run
#
# On astra, prefer SSD space:
#   WAN2GP_ROOT=/mnt/ssd2/zerollama/third_party/wan2gp
# Never binds Gradio to :11434 / :8081. See docs/h3-cuda-port.md.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# Prefer SSD when present (astra); else ~/.zerollama/third_party/wan2gp.
if [[ -z "${WAN2GP_ROOT:-}" ]]; then
  if [[ -d /mnt/ssd2/zerollama/third_party/wan2gp || -d /mnt/ssd2 ]]; then
    WAN2GP_ROOT=/mnt/ssd2/zerollama/third_party/wan2gp
  else
    WAN2GP_ROOT="$HOME/.zerollama/third_party/wan2gp"
  fi
fi
WAN2GP_GIT="${WAN2GP_GIT:-https://github.com/deepbeepmeep/Wan2GP.git}"
HF_REPO="${H3_HF_REPO:-DeepBeepMeep/MiniMax-H3}"

sibling_wan2gp() {
  echo "$(cd "$REPO_ROOT/.." && pwd)/Wan2GP"
}

resolve_wan2gp_repo() {
  if [[ -n "${WAN2GP_REPO:-}" ]]; then
    if [[ -d "$WAN2GP_REPO/.git" || -f "$WAN2GP_REPO/wgp.py" || -d "$WAN2GP_REPO/models/minimax_h3" ]]; then
      echo "$WAN2GP_REPO"
      return
    fi
  fi
  local cand
  for cand in "$(sibling_wan2gp)" /root/Wan2GP "$WAN2GP_ROOT/repo"; do
    if [[ -d "$cand/.git" || -f "$cand/wgp.py" || -d "$cand/models/minimax_h3" ]]; then
      echo "$cand"
      return
    fi
  done
  if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "$(sibling_wan2gp)"
  else
    echo "${WAN2GP_REPO:-$WAN2GP_ROOT/repo}"
  fi
}

ensure_wan2gp_repo() {
  if [[ -d "$WAN2GP_REPO/.git" || -f "$WAN2GP_REPO/wgp.py" || -d "$WAN2GP_REPO/models/minimax_h3" ]]; then
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
# full = FL2VA 33B int8_convrot; pruned = rank8 int8_convrot
VARIANT="${H3_VARIANT:-full}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --weights-only) MODE=weights; shift ;;
    --pruned-only) MODE=weights; VARIANT=pruned; shift ;;
    --full-only) MODE=weights; VARIANT=full; shift ;;
    --venv-only) MODE=venv; shift ;;
    --dry-run) MODE=dryrun; shift ;;
    -h|--help)
      echo "Usage: $0 [--weights-only|--pruned-only|--full-only|--venv-only|--dry-run]"
      echo "Env: WAN2GP_REPO WAN2GP_ROOT WAN2GP_VENV WAN2GP_CKPT_DIR WAN2GP_TORCH_INDEX H3_VARIANT H3_HF_REPO"
      exit 0
      ;;
    \#*) break ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$WAN2GP_ROOT" "$CKPT_DIR/Qwen3-VL-32B-Instruct"
ensure_wan2gp_repo

if [[ ! -d "$WAN2GP_REPO" ]]; then
  echo "Wan2GP repo missing at $WAN2GP_REPO — clone failed; set WAN2GP_REPO" >&2
  exit 1
fi

if [[ "$(cd "$WAN2GP_REPO" && pwd)" != "$(cd "$WAN2GP_ROOT" 2>/dev/null && pwd)/repo" ]]; then
  ln -sfn "$WAN2GP_REPO" "$WAN2GP_ROOT/repo"
  echo "linked $WAN2GP_ROOT/repo -> $WAN2GP_REPO"
fi

if [[ ! -e "$WAN2GP_REPO/ckpts" ]]; then
  ln -sfn "$CKPT_DIR" "$WAN2GP_REPO/ckpts"
  echo "linked $WAN2GP_REPO/ckpts -> $CKPT_DIR"
elif [[ -L "$WAN2GP_REPO/ckpts" ]]; then
  ln -sfn "$CKPT_DIR" "$WAN2GP_REPO/ckpts"
fi

# Symlink ~/.zerollama/third_party -> parent of wan2gp when on SSD.
third_party_parent="$(cd "$WAN2GP_ROOT/.." && pwd)"
if [[ -d "$HOME/.zerollama" ]]; then
  if [[ ! -e "$HOME/.zerollama/third_party" ]]; then
    ln -sfn "$third_party_parent" "$HOME/.zerollama/third_party"
    echo "linked $HOME/.zerollama/third_party -> $third_party_parent"
  fi
fi

dit_file() {
  if [[ "$VARIANT" == "pruned" ]]; then
    echo "MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors"
  else
    echo "MiniMax-H3-FL2VA_int8_convrot.safetensors"
  fi
}

download_weights() {
  local dit te
  dit="$(dit_file)"
  # Pruned lab default: smaller Q2 TE; full prefers Q4 (Wan2GP may still fetch quanto if unset).
  if [[ "$VARIANT" == "pruned" ]]; then
    te="Qwen3-VL-32B-Instruct/qwen3vl-32B-MiniMax-H3-Q2_K.gguf"
  else
    te="Qwen3-VL-32B-Instruct/qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf"
  fi
  python3 - <<PY
from huggingface_hub import hf_hub_download
import os
root = os.path.expanduser("$CKPT_DIR")
os.makedirs(root, exist_ok=True)
os.makedirs(os.path.join(root, "Qwen3-VL-32B-Instruct"), exist_ok=True)
os.makedirs(os.path.join(root, "minimax_h3"), exist_ok=True)
repo = "$HF_REPO"
files = [
    "$dit",
    "minimax_h3_video_vae_fp8mix.safetensors",
    "MiniMax-H3-audio_vae_fp32.safetensors",
    # Wan2GP defaults often pull int8 VAE + latent upscaler under minimax_h3/.
    "minimax_h3/minimax_h3_video_vae_int8_convrot.safetensors",
    "minimax_h3/minimax_h3_latent_upscaler_3d_bf16.safetensors",
    "$te",
    "Qwen3-VL-32B-Instruct/tokenizer.json",
    "Qwen3-VL-32B-Instruct/tokenizer_config.json",
    "Qwen3-VL-32B-Instruct/vocab.json",
    "Qwen3-VL-32B-Instruct/config.json",
    "Qwen3-VL-32B-Instruct/preprocessor_config.json",
    "Qwen3-VL-32B-Instruct/chat_template.json",
]
for f in files:
    print("GET", f, flush=True)
    path = hf_hub_download(repo, f, local_dir=root)
    print("OK", path, flush=True)
print("H3 weights ready under", root, "variant=$VARIANT")
PY
}

check_weights() {
  local missing=0
  local dit
  dit="$(dit_file)"
  local need=(
    "$dit"
    "minimax_h3_video_vae_fp8mix.safetensors"
    "MiniMax-H3-audio_vae_fp32.safetensors"
    "Qwen3-VL-32B-Instruct/qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf"
    "Qwen3-VL-32B-Instruct/tokenizer.json"
  )
  # Accept NVFP4 or Q2 TE as alternatives to Q4.
  for f in "${need[@]}"; do
    if [[ "$f" == *Q4_K_M.gguf ]]; then
      if [[ -f "$CKPT_DIR/$f" ]] \
        || [[ -f "$CKPT_DIR/Qwen3-VL-32B-Instruct/qwen3vl-32B-MiniMax-H3-Q2_K.gguf" ]] \
        || [[ -f "$CKPT_DIR/Qwen3-VL-32B-Instruct/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" ]]; then
        echo "OK text-encoder (Q4/Q2/NVFP4 present)"
        continue
      fi
      echo "MISSING text encoder under $CKPT_DIR/Qwen3-VL-32B-Instruct/"
      missing=1
      continue
    fi
    if [[ ! -f "$CKPT_DIR/$f" ]]; then
      echo "MISSING $CKPT_DIR/$f"
      missing=1
    else
      echo "OK $f ($(du -h "$CKPT_DIR/$f" | awk '{print $1}'))"
    fi
  done
  return "$missing"
}

install_venv() {
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
    pip install torch torchvision torchaudio
  else
    local index="${WAN2GP_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
    pip install "torch" "torchvision" "torchaudio" --index-url "$index" || pip install torch torchvision torchaudio
  fi
  if [[ -f "$WAN2GP_REPO/requirements.txt" ]]; then
    grep -viE '^[[:space:]]*(gradio)([[:space:]]|$)' "$WAN2GP_REPO/requirements.txt" \
      | pip install -r /dev/stdin || true
  fi
  # Wan2GP requirements may pull a mismatched torchaudio (e.g. cu130 vs torch cu128).
  if [[ "$(uname -s)" != "Darwin" ]]; then
    local index="${WAN2GP_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
    pip install --force-reinstall --no-deps "torchaudio" --index-url "$index" || true
  fi
  pip install mmgp huggingface_hub einops || true
  echo "venv ready: $VENV_DIR"
}

run_dry() {
  if [[ "$VARIANT" == "pruned" ]]; then
    export H3_MODEL_TYPE="${H3_MODEL_TYPE:-minimax_h3_fl2va_pruned}"
  else
    export H3_MODEL_TYPE="${H3_MODEL_TYPE:-minimax_h3_fl2va}"
  fi
  check_weights || {
    echo "weights incomplete — run without --dry-run first" >&2
    return 1
  }
  if [[ ! -x "$VENV_DIR/bin/python3" ]]; then
    echo "venv missing — file check only (OK). Install with --venv-only for generate."
    return 0
  fi
  cd "$REPO_ROOT"
  WAN2GP_REPO="$WAN2GP_REPO" WAN2GP_CKPT_DIR="$CKPT_DIR" \
    H3_PROMPT='dry-run probe' H3_SIZE=480x832 H3_FRAMES=17 H3_STEPS=8 \
    H3_MODEL_TYPE="$H3_MODEL_TYPE" \
    H3_OUTPUT_PATH=/tmp/h3-dry.mp4 H3_DRY_RUN=1 \
    "$VENV_DIR/bin/python3" scripts/video/h3_video_generate.py
}

case "$MODE" in
  weights) download_weights; check_weights ;;
  venv) install_venv ;;
  dryrun) run_dry ;;
  all)
    download_weights
    install_venv
    check_weights
    echo "Next: ./scripts/video/register_h3_wan2gp_models.sh"
    echo "Pruned-only: $0 --pruned-only"
    echo "Dry-run: $0 --dry-run  (or H3_DRY_RUN=1 on the wrapper)"
    echo "WAN2GP_ROOT=$WAN2GP_ROOT"
    ;;
esac
