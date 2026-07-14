#!/usr/bin/env bash
# Install Wan2.1 / Wan2.2 text-to-video dependencies and download checkpoints.
# Usage: ./scripts/video/install_wan_video.sh --profile 1.3b|2.2|all
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WAN_ROOT="${WAN_ROOT:-$HOME/.zerollama/third_party/wan}"
VENV_DIR="${WAN_VENV:-$WAN_ROOT/venv}"
PROFILE="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="${2:-all}"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--profile 1.3b|2.2|all]"
      echo ""
      echo "Optional env:"
      echo "  WAN_INSTALL_FLASH_ATTN=1     compile flash_attn (off by default)"
      echo "  WAN_FLASH_ATTN_MAX_JOBS=N    export MAX_JOBS for ninja (default: 1)"
      echo "  WAN_NVCC_THREADS=N           sets NVCC_THREADS per nvcc job (default: 1)"
      echo "  FLASH_ATTN_CUDA_ARCHS=120    limit GPU arches built (default: auto from nvidia-smi)"
      echo "  WAN_TORCH_INDEX=URL          PyTorch wheel index (default: cu128 stable)"
      echo "  WAN_TORCH_PROBE=1            run SM120 cuDNN probe after install (default on)"
      echo "  WAN_DISABLE_CUDNN=0|1        override probe (auto when unset)"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$WAN_ROOT"
if [[ ! -d "$WAN_ROOT/Wan2.1/.git" ]]; then
  git clone --depth 1 --branch main https://github.com/Wan-Video/Wan2.1.git "$WAN_ROOT/Wan2.1"
fi

python3 -m venv "$VENV_DIR"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
pip install -U pip wheel packaging ninja
# torch 2.11 pins setuptools<82; do not blindly upgrade setuptools here.
pip install "setuptools>=70,<82"

# flash_attn setup imports torch; pip build isolation hides torch. Install torch first.
# flash_attn reads MAX_JOBS + NVCC_THREADS (not WAN_*). If MAX_JOBS is unset, setup.py
# auto-sets it from os.cpu_count()//2 (container cores). Wan uses torch SDPA if skipped.
TORCH_INDEX="${WAN_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
pip install "numpy>=1.23.5,<2"
pip install "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" --index-url "$TORCH_INDEX"

install_wan_requirements() {
  local req="$1"
  [[ -f "$req" ]] || return 0
  grep -v '^[[:space:]]*flash_attn[[:space:]]*$' "$req" | pip install -r /dev/stdin
}

install_wan_requirements "$WAN_ROOT/Wan2.1/requirements.txt"
pip install huggingface_hub

patch_wan_attention() {
  local attn="$WAN_ROOT/Wan2.1/wan/modules/attention.py"
  if [[ -f "$attn" ]]; then
    python3 "$REPO_ROOT/scripts/video/patch_wan_attention_sdpa.py" "$attn"
  fi
}
patch_wan_attention

run_wan_torch_probe() {
  if [[ "${WAN_TORCH_PROBE:-1}" == "0" ]]; then
    return 0
  fi
  echo "Probing cuDNN conv on SM120 (5080 class)..."
  if ! "$VENV_DIR/bin/python3" "$REPO_ROOT/scripts/video/wan_torch_compat.py"; then
    echo "note: cuDNN conv failed probe — Wan entry disables cuDNN automatically (native CUDA conv on GPU)." >&2
    echo "      Override: WAN_DISABLE_CUDNN=0 to retry after a torch/cuDNN upgrade." >&2
  fi
}
run_wan_torch_probe

wan_flash_attn_cuda_archs() {
  if [[ -n "${FLASH_ATTN_CUDA_ARCHS:-}" ]]; then
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local cap
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')"
    if [[ -n "$cap" ]]; then
      export FLASH_ATTN_CUDA_ARCHS="$cap"
      return 0
    fi
  fi
  export FLASH_ATTN_CUDA_ARCHS="${WAN_FLASH_ATTN_CUDA_ARCHS:-120}"
}

maybe_install_flash_attn() {
  if [[ "${WAN_INSTALL_FLASH_ATTN:-0}" != "1" ]]; then
    echo "Skipping flash_attn (WAN_INSTALL_FLASH_ATTN=1 to compile; Wan uses torch SDPA)."
    return 0
  fi
  if [[ -z "${CUDA_HOME:-}" ]]; then
    if [[ -d /usr/local/cuda-12.8/bin ]]; then
      export CUDA_HOME=/usr/local/cuda-12.8
    elif [[ -d /usr/local/cuda/bin ]]; then
      export CUDA_HOME=/usr/local/cuda
    fi
  fi
  wan_flash_attn_cuda_archs
  # flash_attn setup.py + torch cpp_extension only honor MAX_JOBS / NVCC_THREADS in the environment.
  export MAX_JOBS="${WAN_FLASH_ATTN_MAX_JOBS:-${MAX_JOBS:-1}}"
  export NVCC_THREADS="${WAN_NVCC_THREADS:-${NVCC_THREADS:-1}}"
  echo "Building flash_attn (MAX_JOBS=${MAX_JOBS}, NVCC_THREADS=${NVCC_THREADS}, FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS})..."
  if ! pip install -v flash_attn --no-build-isolation; then
    echo "warning: flash_attn build failed; continuing without it" >&2
  fi
}
maybe_install_flash_attn

wan_hf_download() {
  local repo="$1" dest="$2"
  if command -v hf >/dev/null 2>&1; then
    hf download "$repo" --local-dir "$dest"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$repo" --local-dir "$dest"
  else
    "$VENV_DIR/bin/python3" -m huggingface_hub.cli download "$repo" --local-dir "$dest"
  fi
}

install_13b() {
  wan_hf_download Wan-AI/Wan2.1-T2V-1.3B "$WAN_ROOT/Wan2.1-T2V-1.3B"
}

install_22() {
  if [[ ! -d "$WAN_ROOT/Wan2.2/.git" ]]; then
    git clone --depth 1 --branch main https://github.com/Wan-Video/Wan2.2.git "$WAN_ROOT/Wan2.2"
  fi
  install_wan_requirements "$WAN_ROOT/Wan2.2/requirements.txt"
  wan_hf_download Wan-AI/Wan2.2-TI2V-5B "$WAN_ROOT/Wan2.2-TI2V-5B"
}

case "$PROFILE" in
  1.3b) install_13b ;;
  2.2) install_22 ;;
  all) install_13b; install_22 ;;
  *) echo "invalid --profile $PROFILE (use 1.3b, 2.2, or all)" >&2; exit 1 ;;
esac

"$VENV_DIR/bin/python3" - <<'PY'
import sys
import torch
print("wan venv ok", sys.executable, "torch", torch.__version__, "cuda", torch.cuda.is_available())
try:
    import flash_attn  # noqa: F401
    print("flash_attn: installed")
except ImportError:
    print("flash_attn: not installed (Wan will use torch SDPA)")
PY

echo "Wan install complete under $WAN_ROOT"
echo "Python venv: $VENV_DIR (set backend_paths.wan_venv or WAN_VENV when running jobs)"
echo "Register models: ./scripts/video/register_wan_models.sh"
