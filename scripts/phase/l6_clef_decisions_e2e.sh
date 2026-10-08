#!/usr/bin/env bash
# Lab e2e: tip llama-server + zerollama /v1/decisions with synth Clef GGUF.
#
# WHY: production CT serve still pins LLAMA_SERVER_BIN at b10615 (no Clef head).
# This starts a lab Go daemon on :11435 with tip b11351 llama-server.
#
# Never binds 11434 / 8080 / 8081.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PORT="${CLEF_E2E_PORT:-11435}"
SRC_GGUF="${CLEF_SMOKE_SRC:-/root/models/tiny-agent/Tiny-Agent-a-0.5B.F16.gguf}"
OUT_GGUF="${CLEF_SMOKE_GGUF:-/tmp/clef-synth-tiny.gguf}"
BIN="${CLEF_SMOKE_BIN:-${ROOT}/vendor/llama-cpp-b11351/build/bin/llama-server}"
# Prefer tip lab binary (System One + ggufIsClef); fall back to production run/zerollama.
if [[ -n "${CLEF_E2E_ZEROLLAMA:-}" ]]; then
  ZL_BIN="${CLEF_E2E_ZEROLLAMA}"
elif [[ -x "${ROOT}/run/zerollama-lab" ]]; then
  ZL_BIN="${ROOT}/run/zerollama-lab"
else
  ZL_BIN="${ROOT}/run/zerollama"
fi
MODEL_NAME="${CLEF_E2E_MODEL:-clef-synth-lab}"
MODELS_DIR="${CLEF_E2E_MODELS:-/tmp/clef-lab-models}"
NUM_GPU="${CLEF_E2E_NUM_GPU:-0}"

case "${PORT}" in
  11434|8080|8081) echo "error: refusing reserved port ${PORT}" >&2; exit 2 ;;
esac
if [[ ! -x "${BIN}" ]]; then
  echo "error: missing tip llama-server ${BIN}" >&2
  exit 1
fi
if [[ ! -x "${ZL_BIN}" ]]; then
  echo "error: missing zerollama ${ZL_BIN}" >&2
  exit 1
fi

if [[ "${CLEF_E2E_SKIP_GRAFT:-0}" == "1" && -f "${OUT_GGUF}" ]]; then
  echo "reusing ${OUT_GGUF} (CLEF_E2E_SKIP_GRAFT=1)"
elif [[ -n "${CLEF_E2E_PRODUCT_GGUF:-}" ]]; then
  OUT_GGUF="${CLEF_E2E_PRODUCT_GGUF}"
  if [[ ! -f "${OUT_GGUF}" ]]; then
    echo "error: CLEF_E2E_PRODUCT_GGUF missing: ${OUT_GGUF}" >&2
    exit 1
  fi
  echo "product GGUF ${OUT_GGUF}"
else
  python3 "${ROOT}/scripts/phase/l6_clef_graft_synth_gguf.py" -i "${SRC_GGUF}" -o "${OUT_GGUF}"
fi

MF="$(mktemp /tmp/clef-synth-XXXXXX.modelfile)"
# CAPABILITY is parsed by newer clients; lab may use an older CLI binary.
# Set decision capability via /api/create after the GGUF import.
cat >"${MF}" <<EOF
FROM ${OUT_GGUF}
RENDERER clef
PARAMETER num_ctx ${CLEF_E2E_NUM_CTX:-512}
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
# Lab e2e must not contend with production VRAM on the 5080 (tip still
# initializes CUDA allocators even with -ngl 0 when devices are visible).
export CUDA_VISIBLE_DEVICES="${CLEF_E2E_CUDA_VISIBLE_DEVICES:-}"

stdbuf -oL -eL "${ZL_BIN}" serve > /tmp/l6_clef_e2e_serve.log 2>&1 &
SPID=$!
cleanup() {
  kill "${SPID}" 2>/dev/null || true
  wait "${SPID}" 2>/dev/null || true
  rm -f "${MF}"
}
trap cleanup EXIT

for i in $(seq 1 90); do
  if ! kill -0 "${SPID}" 2>/dev/null; then
    echo "error: serve died; see /tmp/l6_clef_e2e_serve.log" >&2
    tail -40 /tmp/l6_clef_e2e_serve.log >&2
    exit 1
  fi
  if curl -sS --max-time 2 "http://127.0.0.1:${PORT}/api/version" >/dev/null 2>&1; then
    echo "serve ready (${i}s)"
    break
  fi
  sleep 1
done

OLLAMA_HOST="127.0.0.1:${PORT}" "${ZL_BIN}" create "${MODEL_NAME}" -f "${MF}"

# Attach decision capability + clef renderer (works even when CLI Modelfile lacks CAPABILITY).
curl -sS --max-time 120 "http://127.0.0.1:${PORT}/api/create" \
  -H 'content-type: application/json' \
  -d "{
    \"model\": \"${MODEL_NAME}\",
    \"from\": \"${MODEL_NAME}\",
    \"renderer\": \"clef\",
    \"capabilities\": [\"decision\"],
    \"stream\": false
  }" | tee /tmp/l6_clef_e2e_create_cap.json
echo

curl -sS --max-time 30 "http://127.0.0.1:${PORT}/api/show" \
  -H 'content-type: application/json' \
  -d "{\"model\": \"${MODEL_NAME}\"}" | tee /tmp/l6_clef_e2e_show.json | \
  python3 -c 'import json,sys; d=json.load(sys.stdin); caps=[str(c) for c in (d.get("capabilities") or [])]; print("show caps=", caps, "renderer=", d.get("renderer")); assert "decision" in caps, d'
echo

curl -sS --max-time 300 "http://127.0.0.1:${PORT}/v1/decisions" \
  -H 'content-type: application/json' \
  -d "{
    \"model\": \"${MODEL_NAME}\",
    \"state\": \"Customer charged twice.\",
    \"options\": {\"num_ctx\": 512, \"num_gpu\": ${NUM_GPU}},
    \"questions\": {
      \"refund\": {
        \"type\": \"noul\",
        \"instructions\": \"Refund requested?\",
        \"criteria\": {\"false\": \"no\", \"true\": \"yes\"}
      }
    }
  }" | tee /tmp/l6_clef_e2e_decisions.json

python3 - <<'PY'
import json
d=json.load(open("/tmp/l6_clef_e2e_decisions.json"))
assert "answers" in d or "error" not in d, d
ans=d.get("answers") or {}
assert "refund" in ans, d
print("PASS: l6_clef_decisions_e2e", json.dumps(ans)[:300])
PY
