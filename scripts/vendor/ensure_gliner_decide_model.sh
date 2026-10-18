#!/usr/bin/env bash
# Download fastino/GLiNER2.5-Decide into a local cache dir (stdout: path).
# Used by serve_gliner_decide_lab.sh / smokes. Does not bind serve ports.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_ID="${GLINER_DECIDE_MODEL_ID:-fastino/GLiNER2.5-Decide}"
DEST="${GLINER_DECIDE_MODEL_DIR:-${HOME}/.zerollama/models/GLiNER2.5-Decide}"

if [[ -f "${DEST}/config.json" || -f "${DEST}/model.safetensors" || -f "${DEST}/pytorch_model.bin" ]]; then
  echo "${DEST}"
  exit 0
fi

mkdir -p "${DEST}"
if command -v hf >/dev/null 2>&1; then
  hf download "${MODEL_ID}" --local-dir "${DEST}"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "${MODEL_ID}" --local-dir "${DEST}"
else
  # uv tool / pip one-shot
  if command -v uv >/dev/null 2>&1; then
    uvx --from huggingface_hub hf download "${MODEL_ID}" --local-dir "${DEST}"
  else
    python3 -m pip install -q huggingface_hub
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${MODEL_ID}', local_dir='${DEST}')"
  fi
fi

echo "${DEST}"
