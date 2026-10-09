#!/usr/bin/env bash
# Lab e2e: tip llama-server Strands pointer head + zerollama /v1/decisions.
#
# Never binds 11434 / 8080 / 8081.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PORT="${STRANDS_E2E_PORT:-11436}"
BIN="${STRANDS_SMOKE_BIN:-${ROOT}/vendor/llama-cpp-b11351/build/bin/llama-server}"
GGUF="${STRANDS_E2E_GGUF:-/root/models/strands-hobson-v19-gguf/strands-ollama-q8_0.gguf}"
if [[ -n "${STRANDS_E2E_ZEROLLAMA:-}" ]]; then
  ZL_BIN="${STRANDS_E2E_ZEROLLAMA}"
elif [[ -x "${ROOT}/run/zerollama-lab" ]]; then
  ZL_BIN="${ROOT}/run/zerollama-lab"
else
  ZL_BIN="${ROOT}/run/zerollama"
fi
MODEL_NAME="${STRANDS_E2E_MODEL:-strands-hobson-lab}"
MODELS_DIR="${STRANDS_E2E_MODELS:-/tmp/strands-lab-models}"
NUM_GPU="${STRANDS_E2E_NUM_GPU:-0}"

case "${PORT}" in
  11434|8080|8081) echo "error: refusing reserved port ${PORT}" >&2; exit 2 ;;
esac
[[ -x "${BIN}" ]] || { echo "error: missing tip llama-server ${BIN}" >&2; exit 1; }
[[ -x "${ZL_BIN}" ]] || { echo "error: missing zerollama ${ZL_BIN}" >&2; exit 1; }
[[ -f "${GGUF}" ]] || { echo "error: missing GGUF ${GGUF}" >&2; exit 1; }

MF="$(mktemp /tmp/strands-XXXXXX.modelfile)"
cat >"${MF}" <<EOF
FROM ${GGUF}
RENDERER strands
PARAMETER num_ctx ${STRANDS_E2E_NUM_CTX:-512}
PARAMETER num_gpu ${NUM_GPU}
EOF

mkdir -p "${MODELS_DIR}"
fuser -k "${PORT}/tcp" 2>/dev/null || true
sleep 1

export OLLAMA_HOST="127.0.0.1:${PORT}"
export OLLAMA_MODELS="${MODELS_DIR}"
export LLAMA_SERVER_BIN="${BIN}"
export ZEROLLAMA_LLAMA_SERVER=1
export ZEROLLAMA_RUNTIME_EMBED=0
export ZEROLLAMA_RUNTIME=0
export OLLAMA_MAX_LOADED_MODELS=1
export LD_LIBRARY_PATH="$(dirname "${BIN}"):${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${STRANDS_E2E_CUDA_VISIBLE_DEVICES:-}"

stdbuf -oL -eL "${ZL_BIN}" serve > /tmp/l6_strands_e2e_serve.log 2>&1 &
SPID=$!
cleanup() {
  kill "${SPID}" 2>/dev/null || true
  wait "${SPID}" 2>/dev/null || true
  rm -f "${MF}"
}
trap cleanup EXIT

for i in $(seq 1 90); do
  if ! kill -0 "${SPID}" 2>/dev/null; then
    echo "error: serve died; see /tmp/l6_strands_e2e_serve.log" >&2
    tail -40 /tmp/l6_strands_e2e_serve.log >&2
    exit 1
  fi
  if curl -sS --max-time 2 "http://127.0.0.1:${PORT}/api/version" >/dev/null 2>&1; then
    echo "serve ready (${i}s)"
    break
  fi
  sleep 1
done

OLLAMA_HOST="127.0.0.1:${PORT}" "${ZL_BIN}" create "${MODEL_NAME}" -f "${MF}"

curl -sS --max-time 120 "http://127.0.0.1:${PORT}/api/create" \
  -H 'content-type: application/json' \
  -d "{
    \"model\": \"${MODEL_NAME}\",
    \"from\": \"${MODEL_NAME}\",
    \"renderer\": \"strands\",
    \"capabilities\": [\"decision\"],
    \"stream\": false
  }" | tee /tmp/l6_strands_e2e_create_cap.json
echo

curl -sS --max-time 30 "http://127.0.0.1:${PORT}/api/show" \
  -H 'content-type: application/json' \
  -d "{\"model\": \"${MODEL_NAME}\"}" | tee /tmp/l6_strands_e2e_show.json | \
  python3 -c 'import json,sys; d=json.load(sys.stdin); caps=[str(c) for c in (d.get("capabilities") or [])]; print("show caps=", caps, "renderer=", d.get("renderer")); assert "decision" in caps and d.get("renderer")=="strands", d'
echo

curl -sS --max-time 600 "http://127.0.0.1:${PORT}/v1/decisions" \
  -H 'content-type: application/json' \
  -d "{
    \"model\": \"${MODEL_NAME}\",
    \"state\": \"Customer charged twice for the same order.\",
    \"options\": {\"num_ctx\": 512, \"num_gpu\": ${NUM_GPU}},
    \"questions\": {
      \"refund\": {
        \"type\": \"noul\",
        \"instructions\": \"Refund requested?\",
        \"criteria\": {\"false\": \"no\", \"true\": \"yes\"}
      }
    }
  }" | tee /tmp/l6_strands_e2e_decisions.json

python3 - <<'PY'
import json
d=json.load(open("/tmp/l6_strands_e2e_decisions.json"))
assert "error" not in d, d
ans=d.get("answers") or {}
assert "refund" in ans, d
print("PASS: l6_strands_decisions_e2e", json.dumps(ans)[:400], "usage=", d.get("usage"))
PY
