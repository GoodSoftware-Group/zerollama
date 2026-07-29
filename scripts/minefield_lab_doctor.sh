#!/usr/bin/env bash
# Run upstream model-serving-minefield doctor against a *lab* zerollama.
#
# Never targets production :11434 / :8081. Start lab serve yourself first:
#
#   OLLAMA_HOST=127.0.0.1:11435 \
#   ZEROLLAMA_RUNTIME=0 ZEROLLAMA_RUNTIME_DARWIN_SIDECAR=0 \
#   ./zerollama serve
#
#   ./scripts/minefield_lab_doctor.sh qwen2.5:0.5b
#   ./scripts/minefield_lab_doctor.sh qwen3:0.6b
#
# Env overrides:
#   BASE_URL   default http://127.0.0.1:11435/v1
#   WORK_DIR   default /tmp/zerollama-minefield-lab
set -euo pipefail

MODEL="${1:-}"
if [[ -z "${MODEL}" ]]; then
  echo "usage: $0 <model-tag>" >&2
  exit 2
fi

BASE_URL="${BASE_URL:-http://127.0.0.1:11435/v1}"
WORK_DIR="${WORK_DIR:-/tmp/zerollama-minefield-lab}"
mkdir -p "${WORK_DIR}"

case "${BASE_URL}" in
  *://127.0.0.1:11434*|*://localhost:11434*|*://[::1]:11434*|*://0.0.0.0:11434*)
    echo "refusing production Ollama port in BASE_URL=${BASE_URL}" >&2
    exit 1
    ;;
  *://127.0.0.1:8081*|*://localhost:8081*)
    echo "refusing production runtime port in BASE_URL=${BASE_URL}" >&2
    exit 1
    ;;
esac

DOCTOR="${WORK_DIR}/minefield_doctor.py"
if [[ ! -f "${DOCTOR}" ]]; then
  curl -fsSL -o "${DOCTOR}" \
    https://raw.githubusercontent.com/Blackwellboy/model-serving-minefield/main/doctor/minefield_doctor.py
fi

echo "minefield doctor base=${BASE_URL} model=${MODEL}"
python3 "${DOCTOR}" --base-url "${BASE_URL}" --model "${MODEL}"
