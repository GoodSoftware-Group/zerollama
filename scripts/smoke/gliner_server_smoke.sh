#!/usr/bin/env bash
# GL0/GL1: smoke gliner-server HTTP dual wire (lab ports only).
# Skips cleanly if model or binary missing (CI without ORT/model).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PORT="${GLINER_PORT:-18094}"
MODEL_DIR="${GLINER_MODEL_DIR:-${HOME}/models/gliner_small-v2.1}"
ONNX="${GLINER_ONNX:-${MODEL_DIR}/onnx/model.onnx}"
TOK="${GLINER_TOKENIZER:-${MODEL_DIR}/tokenizer.json}"
BIN="${ZEROLLAMA_GLINER_SERVER_BIN:-}"

if [[ -z "${BIN}" ]]; then
  for c in "${ROOT}/gliner/server/build/gliner-server" "${ROOT}/gliner/server/build/bin/gliner-server"; do
    [[ -x "${c}" ]] && BIN="${c}" && break
  done
fi

if [[ ! -x "${BIN:-}" ]]; then
  echo "skip: gliner-server not built (./scripts/build/build_gliner_server.sh)" >&2
  exit 0
fi
if [[ ! -f "${ONNX}" || ! -f "${TOK}" ]]; then
  echo "skip: model not found at ${ONNX} / ${TOK} (set GLINER_MODEL_DIR)" >&2
  exit 0
fi

if [[ "${PORT}" == "11434" || "${PORT}" == "8081" || "${PORT}" == "8080" ]]; then
  echo "error: refusing production port ${PORT}" >&2
  exit 1
fi

OWNED=0 PID=""
cleanup() { [[ "${OWNED}" == "1" && -n "${PID}" ]] && kill "${PID}" 2>/dev/null || true; }
trap cleanup EXIT

if ! curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  "${BIN}" --model "${ONNX}" --tokenizer "${TOK}" --host 127.0.0.1 --port "${PORT}" \
    > /tmp/gliner-server-smoke.log 2>&1 &
  PID=$!; OWNED=1
  for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
    kill -0 "${PID}" 2>/dev/null || { tail -40 /tmp/gliner-server-smoke.log >&2; exit 1; }
    sleep 1
  done
fi

BODY='{"text":"Kyiv is the capital of Ukraine.","labels":["city","country"],"threshold":0.3}'
echo ">>> /v1/extract" >&2
curl -sf "http://127.0.0.1:${PORT}/v1/extract" -H 'content-type: application/json' -d "${BODY}" | tee /tmp/gliner-extract.json
echo >&2
echo ">>> /v1/gliner" >&2
curl -sf "http://127.0.0.1:${PORT}/v1/gliner" -H 'content-type: application/json' \
  -d '{"text":"Kyiv is the capital of Ukraine.","labels":["city","country"],"threshold":0.3,"flat_ner":true,"max_width":12,"model_type":"span"}' \
  | tee /tmp/gliner-mech.json
echo >&2
python3 - <<'PY'
import json
ex=json.load(open("/tmp/gliner-extract.json"))
gl=json.load(open("/tmp/gliner-mech.json"))
assert "entities" in ex, ex
assert "entities" in gl and "engine" in gl, gl
print("GLINER_SERVER_SMOKE_OK", "extract_n", len(ex["entities"]), "gliner_n", len(gl["entities"]))
PY
