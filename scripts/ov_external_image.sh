#!/usr/bin/env bash
# Wrapper for zerollama openvino-image (OpenVINO GenAI Text2ImagePipeline).
set -euo pipefail

OV_MODEL_DIR="${OLLAMA_OV_MODEL_DIR:-}"
OV_PYTHON="${OLLAMA_OV_PYTHON:-}"
OUTPUT="${OLLAMA_IMAGE_OUTPUT:?OLLAMA_IMAGE_OUTPUT required}"
PROMPT="${OLLAMA_IMAGE_PROMPT:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN="${OLLAMA_OV_GENERATE_PY:-${SCRIPT_DIR}/ov_image_generate.py}"
if [[ ! -f "$GEN" && -f /usr/lib/ollama-zerollama/ov_image_generate.py ]]; then
  GEN=/usr/lib/ollama-zerollama/ov_image_generate.py
fi

if [[ -z "$OV_MODEL_DIR" || ! -d "$OV_MODEL_DIR" ]]; then
  echo "OLLAMA_OV_MODEL_DIR must point to OpenVINO IR weights (install via scripts/install_openvino_diffusion.sh)" >&2
  exit 1
fi
if [[ -z "$PROMPT" ]]; then
  echo "OLLAMA_IMAGE_PROMPT is empty" >&2
  exit 1
fi
if [[ -z "$OV_PYTHON" ]]; then
  if [[ -x /usr/share/zerollama/openvino-genai/venv/bin/python ]]; then
    OV_PYTHON=/usr/share/zerollama/openvino-genai/venv/bin/python
  else
    OV_PYTHON=python3
  fi
fi
if [[ ! -f "$GEN" ]]; then
  echo "missing $GEN" >&2
  exit 1
fi

export OLLAMA_OV_DEVICE="${OLLAMA_OV_DEVICE:-GPU}"

exec "$OV_PYTHON" "$GEN"
