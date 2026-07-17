#!/usr/bin/env bash
# Install OpenVINO GenAI + pre-exported SD models for zerollama openvino-image.
#
# Usage: ./scripts/image/install_openvino_diffusion.sh [--models-only]
#
# Layout (default OV_ROOT=/usr/share/zerollama/openvino-genai):
#   venv/                         Python + openvino + openvino-genai
#   models/sd15-int8-ov/          OpenVINO/stable-diffusion-v1-5-int8-ov
#   models/sdxl-int8-ov/          OpenVINO/stable-diffusion-xl-base-1.0-int8-ov
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OV_ROOT="${OV_ROOT:-/usr/share/zerollama/openvino-genai}"
MODELS_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models-only) MODELS_ONLY=1; shift ;;
    -h|--help)
      echo "Usage: $0 [--models-only]"
      echo "  OV_ROOT=$OV_ROOT"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$OV_ROOT/models"

if [[ "$MODELS_ONLY" -eq 0 ]]; then
  if [[ ! -d "$OV_ROOT/venv" ]]; then
    python3 -m venv "$OV_ROOT/venv"
  fi
  # shellcheck disable=SC1091
  source "$OV_ROOT/venv/bin/activate"
  pip install -U pip wheel
  pip install "openvino>=2024.5.0" "openvino-genai>=2024.5.0" pillow huggingface_hub
  deactivate
fi

download_hf() {
  local repo="$1"
  local dest="$2"
  if [[ -d "$dest" ]] && find "$dest" -name '*.xml' -print -quit | grep -q .; then
    echo "skip (exists): $dest"
    return 0
  fi
  echo "Downloading $repo -> $dest ..."
  "$OV_ROOT/venv/bin/python" - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$repo", local_dir="$dest")
PY
}

# Verify openvino sees GPU (informational)
if [[ -x "$OV_ROOT/venv/bin/python" ]]; then
  "$OV_ROOT/venv/bin/python" - <<'PY' || true
import openvino as ov
core = ov.Core()
print("OpenVINO devices:", core.available_devices)
PY
fi

download_hf "OpenVINO/stable-diffusion-v1-5-int8-ov" "$OV_ROOT/models/sd15-int8-ov"
download_hf "OpenVINO/stable-diffusion-xl-base-1.0-int8-ov" "$OV_ROOT/models/sdxl-int8-ov"

chmod +x "$REPO_ROOT/scripts/image/ov_external_image.sh"

if [[ -d /usr/lib/ollama-zerollama ]]; then
  cp "$REPO_ROOT/scripts/image/ov_external_image.sh" "$REPO_ROOT/scripts/image/ov_image_generate.py" /usr/lib/ollama-zerollama/ 2>/dev/null || \
    sudo cp "$REPO_ROOT/scripts/image/ov_external_image.sh" "$REPO_ROOT/scripts/image/ov_image_generate.py" /usr/lib/ollama-zerollama/
  chmod +x /usr/lib/ollama-zerollama/ov_external_image.sh 2>/dev/null || sudo chmod +x /usr/lib/ollama-zerollama/ov_external_image.sh
fi

echo ""
echo "Installed OpenVINO GenAI under $OV_ROOT"
echo "Register: ./scripts/image/register_ov_models.sh"
echo "Generate: zerollama run sd15-openvino \"a lighthouse at sunset\""
