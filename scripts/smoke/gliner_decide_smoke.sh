#!/usr/bin/env bash
# GD1: GLiNER2.5-Decide sibling smoke (lab ports only).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PORT="${GLINER_DECIDE_PORT:-18098}"
case "${PORT}" in 11434|8081|8080) echo "error: reserved port" >&2; exit 1 ;; esac

MODEL_DIR="$(bash "${ROOT}/scripts/vendor/ensure_gliner_decide_model.sh" | tail -1)"
VENV="${GLINER_DECIDE_VENV:-${ROOT}/gliner/decide_server/.venv}"
APP="${ROOT}/gliner/decide_server/app.py"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo ">>> creating decide venv at ${VENV}" >&2
  python3 -m venv "${VENV}" 2>/dev/null || /root/.local/bin/uv venv "${VENV}"
  "${VENV}/bin/pip" install -q -U pip
  "${VENV}/bin/pip" install -q -r "${ROOT}/gliner/decide_server/requirements.txt"
fi

OWNED=0 PID=""
cleanup() { [[ "${OWNED}" == "1" && -n "${PID}" ]] && kill "${PID}" 2>/dev/null || true; }
trap cleanup EXIT

if ! curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo ">>> starting gliner-decide-server :${PORT}" >&2
  GLINER_DECIDE_MODEL="${MODEL_DIR}" \
  GLINER_DECIDE_DEVICE="${GLINER_DECIDE_DEVICE:-cpu}" \
  GLINER_DECIDE_HOST=127.0.0.1 \
  GLINER_DECIDE_PORT="${PORT}" \
    "${VENV}/bin/python" "${APP}" > /tmp/gliner-decide-smoke.log 2>&1 &
  PID=$!; OWNED=1
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
    kill -0 "${PID}" 2>/dev/null || { tail -60 /tmp/gliner-decide-smoke.log >&2; exit 1; }
    sleep 1
  done
fi

BODY='{"model":"gliner-decide","state":"My subscription renewed after the service was already down. Can I get a refund?","questions":{"intent":{"type":"choice","instructions":"intent","criteria":{"refund_request":"wants refund","cancel_subscription":"cancel","order_status":"tracking","other":"other"}}}}'
curl -sf "http://127.0.0.1:${PORT}/v1/systemone" -H 'content-type: application/json' -d "${BODY}" \
  | tee /tmp/gliner-decide-systemone.json
echo >&2

MECH='{"model":"gliner-decide","text":"My subscription renewed after the service was already down. Can I get a refund?","schema":{"intent":["refund_request","cancel_subscription","order_status","other"]},"include_confidence":true}'
curl -sf "http://127.0.0.1:${PORT}/v1/gliner-decide" -H 'content-type: application/json' -d "${MECH}" \
  | tee /tmp/gliner-decide-mech.json
echo >&2

python3 - <<'PY'
import json
doc=json.load(open("/tmp/gliner-decide-systemone.json"))
assert "answers" in doc and "intent" in doc["answers"], doc
ans=doc["answers"]["intent"]
assert ans.get("type")=="choice", ans
assert ans.get("calibrated") is False, ans
choice=ans.get("choice") or ""
assert choice, ans
mech=json.load(open("/tmp/gliner-decide-mech.json"))
assert "result" in mech, mech
print("GLINER_DECIDE_SMOKE_OK", "choice", choice, "mech_keys", list((mech.get("result") or {}).keys()))
PY
