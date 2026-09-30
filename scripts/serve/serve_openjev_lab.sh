#!/usr/bin/env bash
# OpenJev / DiffusionGemma sibling serve (DG8/DG9).
# Lab/production-capable Decider path — NEVER binds :11434 / :8081 / :8080.
# Sibling diffusion on :18093; Go OpenJev proxy on :11436 (override via env).
#
# Usage:
#   ./scripts/serve/serve_openjev_lab.sh                         # uncalibrated
#   ./scripts/serve/serve_openjev_lab.sh --calibrated             # choice/score calibrated:true
#   ./scripts/serve/serve_openjev_lab.sh --calibrated --noul-calibrated  # + noul (DG9)
#
# Free emb/CLM VRAM on the 5080 before -ngl 25. Default DIFFUSION_NGL=8 when tight.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=refuse_pve_host.sh
source "${ROOT}/scripts/serve/refuse_pve_host.sh"

CALIBRATED=0
NOUL_CAL=0
for arg in "$@"; do
  case "${arg}" in
    --calibrated) CALIBRATED=1 ;;
    --noul-calibrated) NOUL_CAL=1 ;;
    -h|--help)
      sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
  esac
done

export LD_LIBRARY_PATH="/root/nvidia-host:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/go/bin:${PATH:-}"

DIFF_PORT="${DIFFUSION_PORT:-18093}"
GO_HOST="${ZEROLLAMA_OPENJEV_HOST:-127.0.0.1:11436}"
MODEL="${DIFFUSION_GGUF:-/root/models/diffusiongemma/diffusiongemma-26B-A4B-it-Q4_K_M.gguf}"
NGL="${DIFFUSION_NGL:-8}"

if [[ -z "${ZEROLLAMA_DIFFUSION_SERVER_BIN:-}" ]]; then
  CAND="${ROOT}/vendor/llama-cpp-diffusion-dd0cf0445/build/bin/llama-diffusion-gemma-server"
  [[ -x "${CAND}" ]] || { echo "error: build sibling first (scripts/build/build_llama_diffusion.sh)" >&2; exit 1; }
  ZEROLLAMA_DIFFUSION_SERVER_BIN="${CAND}"
fi
ZL="${ROOT}/zerollama"
[[ -x "${ZL}" ]] || { echo "error: missing ${ZL} (go build -o zerollama .)" >&2; exit 1; }
[[ -f "${MODEL}" ]] || { echo "error: missing GGUF ${MODEL}" >&2; exit 1; }

# Refuse reserved production ports.
GO_PORT="${GO_HOST##*:}"
case "${DIFF_PORT}" in 11434|8081|8080) echo "error: DIFFUSION_PORT=${DIFF_PORT} is reserved" >&2; exit 1 ;; esac
case "${GO_PORT}" in 11434|8081|8080) echo "error: Go port ${GO_PORT} is reserved" >&2; exit 1 ;; esac

DIFF_PID="" GO_PID=""
cleanup() {
  [[ -n "${GO_PID}" ]] && kill "${GO_PID}" 2>/dev/null || true
  [[ -n "${DIFF_PID}" ]] && kill "${DIFF_PID}" 2>/dev/null || true
}
trap cleanup EXIT

if ! curl -sf "http://127.0.0.1:${DIFF_PORT}/props" >/dev/null 2>&1; then
  echo ">>> diffusion :${DIFF_PORT} -ngl ${NGL}"
  "${ZEROLLAMA_DIFFUSION_SERVER_BIN}" -m "${MODEL}" -ngl "${NGL}" -c 2048 \
    --host 127.0.0.1 --port "${DIFF_PORT}" > /tmp/openjev-diff.log 2>&1 &
  DIFF_PID=$!
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:${DIFF_PORT}/props" >/dev/null 2>&1 && break
    kill -0 "${DIFF_PID}" 2>/dev/null || { tail -40 /tmp/openjev-diff.log >&2; exit 1; }
    sleep 2
  done
else
  echo ">>> reusing diffusion :${DIFF_PORT}"
fi

CAL_ENV=()
if [[ "${CALIBRATED}" == "1" ]]; then
  CAL_ENV+=(ZEROLLAMA_OPENJEV_CALIBRATED=1)
  echo ">>> DG8 calibrated opt-in (choice/score)"
fi
if [[ "${NOUL_CAL}" == "1" ]]; then
  CAL_ENV+=(ZEROLLAMA_OPENJEV_NOUL_CALIBRATED=1)
  echo ">>> DG9 noul calibrated opt-in (requires --calibrated for answers to flip)"
fi
if [[ -n "${ZEROLLAMA_OPENJEV_NOUL_BIAS:-}" ]]; then
  CAL_ENV+=(ZEROLLAMA_OPENJEV_NOUL_BIAS="${ZEROLLAMA_OPENJEV_NOUL_BIAS}")
fi

echo ">>> zerollama OpenJev proxy ${GO_HOST}"
env OLLAMA_HOST="${GO_HOST}" \
  ZEROLLAMA_OPENJEV_URL="http://127.0.0.1:${DIFF_PORT}" \
  OLLAMA_NO_CLOUD=true OLLAMA_TRAINING=false ZEROLLAMA_RUNTIME_EMBED=0 \
  "${CAL_ENV[@]}" \
  "${ZL}" serve > /tmp/openjev-go.log 2>&1 &
GO_PID=$!
for _ in $(seq 1 60); do
  curl -sf "http://${GO_HOST}/api/version" >/dev/null 2>&1 && break
  kill -0 "${GO_PID}" 2>/dev/null || { tail -40 /tmp/openjev-go.log >&2; exit 1; }
  sleep 1
done

echo "OpenJev ready: http://${GO_HOST}/v1/systemone  (model=openjev)"
echo "Logs: /tmp/openjev-diff.log /tmp/openjev-go.log"
echo "Ctrl-C stops owned processes."
wait
