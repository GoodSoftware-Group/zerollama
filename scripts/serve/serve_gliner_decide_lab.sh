#!/usr/bin/env bash
# GLiNER2.5-Decide sibling + Go proxy (GD1/GD3). NEVER binds :11434 / :8081 / :8080.
#
# Usage:
#   ./scripts/serve/serve_gliner_decide_lab.sh
#   GLINER_DECIDE_DEVICE=cuda:0 ./scripts/serve/serve_gliner_decide_lab.sh
#
# Sibling :18098. Go :11439.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=refuse_pve_host.sh
source "${ROOT}/scripts/serve/refuse_pve_host.sh"

export PATH="/usr/local/go/bin:${PATH:-}"
DEC_PORT="${GLINER_DECIDE_PORT:-18098}"
GO_HOST="${ZEROLLAMA_GLINER_DECIDE_HOST:-127.0.0.1:11439}"
DEVICE="${GLINER_DECIDE_DEVICE:-cpu}"
MODEL_DIR="$(bash "${ROOT}/scripts/vendor/ensure_gliner_decide_model.sh" | tail -1)"
VENV="${GLINER_DECIDE_VENV:-${ROOT}/gliner/decide_server/.venv}"
APP="${ROOT}/gliner/decide_server/app.py"
ZL="${ROOT}/zerollama"

[[ -x "${ZL}" ]] || { echo "error: missing ${ZL}" >&2; exit 1; }
if [[ ! -x "${VENV}/bin/python" ]]; then
  echo ">>> creating decide venv" >&2
  python3 -m venv "${VENV}" 2>/dev/null || /root/.local/bin/uv venv "${VENV}"
  "${VENV}/bin/pip" install -q -U pip
  "${VENV}/bin/pip" install -q -r "${ROOT}/gliner/decide_server/requirements.txt"
fi

GO_PORT="${GO_HOST##*:}"
case "${DEC_PORT}" in 11434|8081|8080) echo "error: DEC port reserved" >&2; exit 1 ;; esac
case "${GO_PORT}" in 11434|8081|8080) echo "error: Go port reserved" >&2; exit 1 ;; esac

DEC_PID="" GO_PID=""
cleanup() {
  [[ -n "${GO_PID}" ]] && kill "${GO_PID}" 2>/dev/null || true
  [[ -n "${DEC_PID}" ]] && kill "${DEC_PID}" 2>/dev/null || true
}
trap cleanup EXIT

if ! curl -sf "http://127.0.0.1:${DEC_PORT}/health" >/dev/null 2>&1; then
  echo ">>> gliner-decide-server :${DEC_PORT} device=${DEVICE}"
  GLINER_DECIDE_MODEL="${MODEL_DIR}" GLINER_DECIDE_DEVICE="${DEVICE}" \
  GLINER_DECIDE_HOST=127.0.0.1 GLINER_DECIDE_PORT="${DEC_PORT}" \
    "${VENV}/bin/python" "${APP}" > /tmp/gliner-decide-lab-server.log 2>&1 &
  DEC_PID=$!
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:${DEC_PORT}/health" >/dev/null 2>&1 && break
    kill -0 "${DEC_PID}" 2>/dev/null || { tail -40 /tmp/gliner-decide-lab-server.log >&2; exit 1; }
    sleep 1
  done
else
  echo ">>> reusing gliner-decide-server :${DEC_PORT}"
fi

echo ">>> zerollama Decide proxy ${GO_HOST}"
OLLAMA_HOST="${GO_HOST}" \
  ZEROLLAMA_GLINER_DECIDE_URL="http://127.0.0.1:${DEC_PORT}" \
  OLLAMA_NO_CLOUD=true OLLAMA_TRAINING=false ZEROLLAMA_RUNTIME_EMBED=0 \
  "${ZL}" serve > /tmp/gliner-decide-lab-go.log 2>&1 &
GO_PID=$!
for _ in $(seq 1 60); do
  curl -sf "http://${GO_HOST}/api/version" >/dev/null 2>&1 && break
  kill -0 "${GO_PID}" 2>/dev/null || { tail -40 /tmp/gliner-decide-lab-go.log >&2; exit 1; }
  sleep 1
done

echo "GLiNER2.5-Decide ready:"
echo "  http://${GO_HOST}/v1/decisions     (abstract Jev)"
echo "  http://${GO_HOST}/v1/gliner-decide (mechanical)"
echo "Ctrl+C stops both."
wait
