#!/usr/bin/env bash
# GLiNER sibling + Go proxy (GL2/GL4). NEVER binds :11434 / :8081 / :8080.
#
# Usage:
#   ./scripts/serve/serve_gliner_lab.sh              # CPU ORT, span small
#   ./scripts/serve/serve_gliner_lab.sh --cuda        # CUDA ORT --device-id 0
#   ./scripts/serve/serve_gliner_lab.sh --multitask   # token-level multitask model (GL5a)
#
# Sibling :18094 (CUDA smoke may use :18095 via GLINER_PORT). Go :11437.
# Free OpenJev/emb VRAM before --cuda on 16 GB (Finding 7).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=refuse_pve_host.sh
source "${ROOT}/scripts/serve/refuse_pve_host.sh"

CUDA=0
MULTITASK=0
for arg in "$@"; do
  case "${arg}" in
    --cuda) CUDA=1 ;;
    --multitask|--token) MULTITASK=1 ;;
    -h|--help)
      sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
  esac
done

export PATH="/usr/local/go/bin:${PATH:-}"
GL_PORT="${GLINER_PORT:-18094}"
GO_HOST="${ZEROLLAMA_GLINER_HOST:-127.0.0.1:11437}"
DEVICE_ARGS=()
MODEL_TYPE=span

MAX_WIDTH=12
MAX_LENGTH=512
if [[ "${MULTITASK}" == "1" ]]; then
  MODEL_DIR="${GLINER_MODEL_DIR:-${HOME}/models/gliner-multitask-large-v0.5}"
  MODEL_TYPE=token
  # Upstream token example uses 12/512; max_width unused by TokenProcessor.
  # max_length 768 matches gliner_config max_len for long prompts.
  MAX_WIDTH=12
  MAX_LENGTH=768
  bash "${ROOT}/scripts/vendor/ensure_gliner_model.sh" multitask >/dev/null
else
  MODEL_DIR="${GLINER_MODEL_DIR:-${HOME}/models/gliner_small-v2.1}"
  MODEL_TYPE=span
  bash "${ROOT}/scripts/vendor/ensure_gliner_model.sh" small >/dev/null
fi
ONNX="${GLINER_ONNX:-${MODEL_DIR}/onnx/model.onnx}"
TOK="${GLINER_TOKENIZER:-${MODEL_DIR}/tokenizer.json}"

if [[ -z "${ZEROLLAMA_GLINER_SERVER_BIN:-}" ]]; then
  if [[ "${CUDA}" == "1" ]]; then
    CAND="${ROOT}/gliner/server/build-cuda/gliner-server"
  else
    CAND="${ROOT}/gliner/server/build/gliner-server"
  fi
  [[ -x "${CAND}" ]] || { echo "error: build gliner-server first (GLINER_ORT=cuda optional)" >&2; exit 1; }
  ZEROLLAMA_GLINER_SERVER_BIN="${CAND}"
fi
ZL="${ROOT}/zerollama"
[[ -x "${ZL}" ]] || { echo "error: missing ${ZL} (go build -o zerollama .)" >&2; exit 1; }
[[ -f "${ONNX}" && -f "${TOK}" ]] || { echo "error: missing model under ${MODEL_DIR}" >&2; exit 1; }

GO_PORT="${GO_HOST##*:}"
case "${GL_PORT}" in 11434|8081|8080) echo "error: GLINER_PORT reserved" >&2; exit 1 ;; esac
case "${GO_PORT}" in 11434|8081|8080) echo "error: Go port reserved" >&2; exit 1 ;; esac

export LD_LIBRARY_PATH="/root/nvidia-host:${LD_LIBRARY_PATH:-}"
if [[ "${CUDA}" == "1" ]]; then
  ORT_GPU="${ROOT}/vendor/onnxruntime-linux-x64-gpu-1.19.2"
  [[ -d "${ORT_GPU}/lib" ]] && export LD_LIBRARY_PATH="${ORT_GPU}/lib:${LD_LIBRARY_PATH}"
  for d in /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib /usr/lib/ollama/mlx_cuda_v12; do
    [[ -d "${d}" ]] && export LD_LIBRARY_PATH="${d}:${LD_LIBRARY_PATH}"
  done
  DEVICE_ARGS+=(--device-id "${GLINER_DEVICE_ID:-0}")
  echo ">>> CUDA device_id=${GLINER_DEVICE_ID:-0} (unload OpenJev/emb first on 16GB)"
fi

GL_PID="" GO_PID=""
cleanup() {
  [[ -n "${GO_PID}" ]] && kill "${GO_PID}" 2>/dev/null || true
  [[ -n "${GL_PID}" ]] && kill "${GL_PID}" 2>/dev/null || true
}
trap cleanup EXIT

if ! curl -sf "http://127.0.0.1:${GL_PORT}/health" >/dev/null 2>&1; then
  echo ">>> gliner-server :${GL_PORT} model_type=${MODEL_TYPE}"
  "${ZEROLLAMA_GLINER_SERVER_BIN}" \
    --model "${ONNX}" --tokenizer "${TOK}" \
    --host 127.0.0.1 --port "${GL_PORT}" \
    --model-type "${MODEL_TYPE}" \
    --max-width "${MAX_WIDTH}" --max-length "${MAX_LENGTH}" \
    "${DEVICE_ARGS[@]}" \
    > /tmp/gliner-lab-server.log 2>&1 &
  GL_PID=$!
  for _ in $(seq 1 90); do
    curl -sf "http://127.0.0.1:${GL_PORT}/health" >/dev/null 2>&1 && break
    kill -0 "${GL_PID}" 2>/dev/null || { tail -40 /tmp/gliner-lab-server.log >&2; exit 1; }
    sleep 1
  done
else
  echo ">>> reusing gliner-server :${GL_PORT}"
fi

echo ">>> zerollama GLiNER proxy ${GO_HOST}"
OLLAMA_HOST="${GO_HOST}" \
  ZEROLLAMA_GLINER_URL="http://127.0.0.1:${GL_PORT}" \
  OLLAMA_NO_CLOUD=true OLLAMA_TRAINING=false ZEROLLAMA_RUNTIME_EMBED=0 \
  "${ZL}" serve > /tmp/gliner-lab-go.log 2>&1 &
GO_PID=$!
for _ in $(seq 1 60); do
  curl -sf "http://${GO_HOST}/api/version" >/dev/null 2>&1 && break
  kill -0 "${GO_PID}" 2>/dev/null || { tail -40 /tmp/gliner-lab-go.log >&2; exit 1; }
  sleep 1
done

echo "GLiNER ready:"
echo "  http://${GO_HOST}/v1/extract   (abstract)"
echo "  http://${GO_HOST}/v1/gliner    (mechanical; model_type=${MODEL_TYPE})"
echo "Ctrl+C stops both."
wait
