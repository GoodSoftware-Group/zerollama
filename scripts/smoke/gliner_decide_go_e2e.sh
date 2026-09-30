#!/usr/bin/env bash
# GD2: Go :11439 → gliner-decide-server :18098 (lab only).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GO_PORT="${ZEROLLAMA_LAB_PORT:-11439}"
DEC_PORT="${GLINER_DECIDE_PORT:-18098}"
ZL="${ROOT}/zerollama"
[[ -x "${ZL}" ]] || ZL="$(command -v zerollama || true)"
VENV="${GLINER_DECIDE_VENV:-${ROOT}/gliner/decide_server/.venv}"
APP="${ROOT}/gliner/decide_server/app.py"
MODEL_DIR="$(bash "${ROOT}/scripts/vendor/ensure_gliner_decide_model.sh" | tail -1)"

for p in "${GO_PORT}" "${DEC_PORT}"; do
  case "${p}" in 11434|8081|8080) echo "error: reserved port ${p}" >&2; exit 1 ;; esac
done
if [[ ! -x "${ZL:-}" ]]; then
  echo "skip: need zerollama binary" >&2
  exit 0
fi
if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "skip: decide venv missing (run gliner_decide_smoke.sh first)" >&2
  exit 0
fi

OWNED_DEC=0 OWNED_GO=0 DEC_PID="" GO_PID=""
cleanup() {
  [[ "${OWNED_GO}" == "1" && -n "${GO_PID}" ]] && kill "${GO_PID}" 2>/dev/null || true
  [[ "${OWNED_DEC}" == "1" && -n "${DEC_PID}" ]] && kill "${DEC_PID}" 2>/dev/null || true
}
trap cleanup EXIT

if ! curl -sf "http://127.0.0.1:${DEC_PORT}/health" >/dev/null 2>&1; then
  GLINER_DECIDE_MODEL="${MODEL_DIR}" GLINER_DECIDE_DEVICE="${GLINER_DECIDE_DEVICE:-cpu}" \
  GLINER_DECIDE_HOST=127.0.0.1 GLINER_DECIDE_PORT="${DEC_PORT}" \
    "${VENV}/bin/python" "${APP}" > /tmp/gliner-decide-e2e-dec.log 2>&1 &
  DEC_PID=$!; OWNED_DEC=1
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:${DEC_PORT}/health" >/dev/null 2>&1 && break
    kill -0 "${DEC_PID}" 2>/dev/null || { tail -40 /tmp/gliner-decide-e2e-dec.log >&2; exit 1; }
    sleep 1
  done
fi

if ! curl -sf "http://127.0.0.1:${GO_PORT}/api/version" >/dev/null 2>&1; then
  OLLAMA_HOST="127.0.0.1:${GO_PORT}" \
  ZEROLLAMA_GLINER_DECIDE_URL="http://127.0.0.1:${DEC_PORT}" \
  OLLAMA_NO_CLOUD=true OLLAMA_TRAINING=false ZEROLLAMA_RUNTIME_EMBED=0 \
    "${ZL}" serve > /tmp/gliner-decide-e2e-go.log 2>&1 &
  GO_PID=$!; OWNED_GO=1
  for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:${GO_PORT}/api/version" >/dev/null 2>&1 && break
    kill -0 "${GO_PID}" 2>/dev/null || { tail -40 /tmp/gliner-decide-e2e-go.log >&2; exit 1; }
    sleep 1
  done
fi

curl -sf "http://127.0.0.1:${GO_PORT}/v1/decisions" -H 'content-type: application/json' \
  -d '{"model":"gliner-decide","state":"My subscription renewed after the service was already down. Can I get a refund?","questions":{"intent":{"type":"choice","instructions":"intent","criteria":{"refund_request":"refund","cancel_subscription":"cancel","other":"other"}}}}' \
  | tee /tmp/gliner-decide-e2e-decisions.json
echo >&2
curl -sf "http://127.0.0.1:${GO_PORT}/v1/gliner-decide" -H 'content-type: application/json' \
  -d '{"model":"gliner-decide","text":"My subscription renewed after the service was already down. Can I get a refund?","schema":{"intent":["refund_request","cancel_subscription","other"]},"include_confidence":true}' \
  | tee /tmp/gliner-decide-e2e-mech.json
echo >&2
python3 - <<'PY'
import json
d=json.load(open("/tmp/gliner-decide-e2e-decisions.json"))
m=json.load(open("/tmp/gliner-decide-e2e-mech.json"))
ans=d["answers"]["intent"]
assert ans["type"]=="choice" and ans.get("choice"), ans
assert ans.get("calibrated") is False
assert "result" in m
print("GLINER_DECIDE_GO_E2E_OK", "choice", ans["choice"])
PY
