#!/usr/bin/env bash
# GL2: Go :11437 → gliner-server :18094 (lab only). Skips if binary/model missing.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GO_PORT="${ZEROLLAMA_LAB_PORT:-11437}"
GL_PORT="${GLINER_PORT:-18094}"
MODEL_DIR="${GLINER_MODEL_DIR:-${HOME}/models/gliner_small-v2.1}"
ONNX="${GLINER_ONNX:-${MODEL_DIR}/onnx/model.onnx}"
TOK="${GLINER_TOKENIZER:-${MODEL_DIR}/tokenizer.json}"
BIN="${ZEROLLAMA_GLINER_SERVER_BIN:-}"
ZL="${ROOT}/zerollama"
[[ -x "${ZL}" ]] || ZL="$(command -v zerollama || true)"

if [[ -z "${BIN}" ]]; then
  for c in "${ROOT}/gliner/server/build/gliner-server" "${ROOT}/gliner/server/build/bin/gliner-server"; do
    [[ -x "${c}" ]] && BIN="${c}" && break
  done
fi
if [[ ! -x "${BIN:-}" || ! -f "${ONNX}" || ! -x "${ZL:-}" ]]; then
  echo "skip: need gliner-server + model + zerollama binary" >&2
  exit 0
fi
for p in "${GO_PORT}" "${GL_PORT}"; do
  if [[ "${p}" == "11434" || "${p}" == "8081" || "${p}" == "8080" ]]; then
    echo "error: refusing production port ${p}" >&2
    exit 1
  fi
done

OWNED_GL=0 OWNED_GO=0 GL_PID="" GO_PID=""
cleanup() {
  [[ "${OWNED_GO}" == "1" && -n "${GO_PID}" ]] && kill "${GO_PID}" 2>/dev/null || true
  [[ "${OWNED_GL}" == "1" && -n "${GL_PID}" ]] && kill "${GL_PID}" 2>/dev/null || true
}
trap cleanup EXIT

if ! curl -sf "http://127.0.0.1:${GL_PORT}/health" >/dev/null 2>&1; then
  "${BIN}" --model "${ONNX}" --tokenizer "${TOK}" --host 127.0.0.1 --port "${GL_PORT}" \
    > /tmp/gliner-e2e-gl.log 2>&1 &
  GL_PID=$!; OWNED_GL=1
  for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:${GL_PORT}/health" >/dev/null 2>&1 && break
    kill -0 "${GL_PID}" 2>/dev/null || { tail -40 /tmp/gliner-e2e-gl.log >&2; exit 1; }
    sleep 1
  done
fi

if ! curl -sf "http://127.0.0.1:${GO_PORT}/api/version" >/dev/null 2>&1; then
  OLLAMA_HOST="127.0.0.1:${GO_PORT}" \
  ZEROLLAMA_GLINER_URL="http://127.0.0.1:${GL_PORT}" \
  OLLAMA_NO_CLOUD=true OLLAMA_TRAINING=false ZEROLLAMA_RUNTIME_EMBED=0 \
    "${ZL}" serve > /tmp/gliner-e2e-go.log 2>&1 &
  GO_PID=$!; OWNED_GO=1
  for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:${GO_PORT}/api/version" >/dev/null 2>&1 && break
    kill -0 "${GO_PID}" 2>/dev/null || { tail -40 /tmp/gliner-e2e-go.log >&2; exit 1; }
    sleep 1
  done
fi

curl -sf "http://127.0.0.1:${GO_PORT}/v1/extract" -H 'content-type: application/json' \
  -d '{"model":"gliner","text":"Kyiv is the capital of Ukraine.","labels":["city","country"],"threshold":0.3,"max_width":99}' \
  | tee /tmp/gliner-e2e-extract.json
echo >&2
curl -sf "http://127.0.0.1:${GO_PORT}/v1/gliner" -H 'content-type: application/json' \
  -d '{"model":"gliner","text":"Kyiv is the capital of Ukraine.","labels":["city","country"],"threshold":0.3,"flat_ner":true,"max_width":12}' \
  | tee /tmp/gliner-e2e-gliner.json
echo >&2
python3 - <<'PY'
import json
ex=json.load(open("/tmp/gliner-e2e-extract.json"))
gl=json.load(open("/tmp/gliner-e2e-gliner.json"))
assert "entities" in ex
assert "entities" in gl and "engine" in gl
print("GLINER_GO_E2E_OK")
PY
