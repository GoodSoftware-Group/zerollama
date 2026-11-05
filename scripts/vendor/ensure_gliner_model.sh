#!/usr/bin/env bash
# Download ONNX GLiNER models for lab smokes (GL0/GL5a).
#
# Usage:
#   ./scripts/vendor/ensure_gliner_model.sh                 # gliner_small-v2.1
#   ./scripts/vendor/ensure_gliner_model.sh small
#   ./scripts/vendor/ensure_gliner_model.sh multitask        # token-level multitask (GL5a)
#
# Env:
#   GLINER_MODEL_DIR — override install dir
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KIND="${1:-small}"

case "${KIND}" in
  small|v2.1|gliner_small)
    REPO="onnx-community/gliner_small-v2.1"
    DEST="${GLINER_MODEL_DIR:-${HOME}/models/gliner_small-v2.1}"
    ;;
  multitask|token|multitask-large)
    REPO="onnx-community/gliner-multitask-large-v0.5"
    DEST="${GLINER_MODEL_DIR:-${HOME}/models/gliner-multitask-large-v0.5}"
    ;;
  -h|--help)
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  *)
    echo "error: unknown model kind '${KIND}' (small|multitask)" >&2
    exit 1
    ;;
esac

if [[ -f "${DEST}/onnx/model.onnx" && -f "${DEST}/tokenizer.json" ]]; then
  echo ">>> already have ${DEST}" >&2
  echo "${DEST}"
  exit 0
fi

mkdir -p "${DEST}"
echo ">>> snapshot_download ${REPO} → ${DEST}" >&2
python3 - <<PY
from huggingface_hub import snapshot_download
import os
snapshot_download(
    "${REPO}",
    local_dir=os.environ.get("DEST", "${DEST}"),
    allow_patterns=["onnx/*.onnx", "tokenizer.json", "tokenizer_config.json", "config.json", "gliner_config.json", "*.json"],
)
print("ok")
PY

[[ -f "${DEST}/onnx/model.onnx" && -f "${DEST}/tokenizer.json" ]] \
  || { echo "error: download incomplete under ${DEST}" >&2; exit 1; }
echo "${DEST}"
