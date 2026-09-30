#!/usr/bin/env bash
# GL5a: Go proxy → token multitask gliner-server (lab only).
# Ports: sibling :18097, Go :11438 (never 11434/8081/8080).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GO_PORT="${ZEROLLAMA_LAB_PORT:-11438}"
GL_PORT="${GLINER_PORT:-18097}"
MODEL_DIR="$(bash "${ROOT}/scripts/vendor/ensure_gliner_model.sh" multitask | tail -1)"
ONNX="${MODEL_DIR}/onnx/model.onnx"
TOK="${MODEL_DIR}/tokenizer.json"
BIN="${ZEROLLAMA_GLINER_SERVER_BIN:-}"
ZL="${ROOT}/zerollama"
[[ -x "${ZL}" ]] || ZL="$(command -v zerollama || true)"

if [[ -z "${BIN}" ]]; then
  for c in "${ROOT}/gliner/server/build/gliner-server" "${ROOT}/gliner/server/build-cuda/gliner-server"; do
    [[ -x "${c}" ]] && BIN="${c}" && break
  done
fi
if [[ ! -x "${BIN:-}" || ! -f "${ONNX}" || ! -x "${ZL:-}" ]]; then
  echo "skip: need gliner-server + multitask model + zerollama" >&2
  exit 0
fi
for p in "${GO_PORT}" "${GL_PORT}"; do
  case "${p}" in 11434|8081|8080) echo "error: reserved port ${p}" >&2; exit 1 ;; esac
done

ORT_CPU="${ROOT}/vendor/onnxruntime-linux-x64-1.19.2"
export LD_LIBRARY_PATH="${ORT_CPU}/lib:/root/nvidia-host:${LD_LIBRARY_PATH:-}"

OWNED_GL=0 OWNED_GO=0 GL_PID="" GO_PID=""
cleanup() {
  [[ "${OWNED_GO}" == "1" && -n "${GO_PID}" ]] && kill "${GO_PID}" 2>/dev/null || true
  [[ "${OWNED_GL}" == "1" && -n "${GL_PID}" ]] && kill "${GL_PID}" 2>/dev/null || true
}
trap cleanup EXIT

if ! curl -sf "http://127.0.0.1:${GL_PORT}/health" >/dev/null 2>&1; then
  "${BIN}" --model "${ONNX}" --tokenizer "${TOK}" \
    --host 127.0.0.1 --port "${GL_PORT}" --model-type token \
    --max-width 12 --max-length 768 \
    > /tmp/gliner-token-e2e-gl.log 2>&1 &
  GL_PID=$!; OWNED_GL=1
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:${GL_PORT}/health" >/dev/null 2>&1 && break
    kill -0 "${GL_PID}" 2>/dev/null || { tail -40 /tmp/gliner-token-e2e-gl.log >&2; exit 1; }
    sleep 1
  done
fi

if ! curl -sf "http://127.0.0.1:${GO_PORT}/api/version" >/dev/null 2>&1; then
  OLLAMA_HOST="127.0.0.1:${GO_PORT}" \
  ZEROLLAMA_GLINER_URL="http://127.0.0.1:${GL_PORT}" \
  OLLAMA_NO_CLOUD=true OLLAMA_TRAINING=false ZEROLLAMA_RUNTIME_EMBED=0 \
    "${ZL}" serve > /tmp/gliner-token-e2e-go.log 2>&1 &
  GO_PID=$!; OWNED_GO=1
  for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:${GO_PORT}/api/version" >/dev/null 2>&1 && break
    kill -0 "${GO_PID}" 2>/dev/null || { tail -40 /tmp/gliner-token-e2e-go.log >&2; exit 1; }
    sleep 1
  done
fi

curl -sf "http://127.0.0.1:${GO_PORT}/v1/extract" -H 'content-type: application/json' \
  -d '{"model":"gliner","text":"Kyiv is the capital of Ukraine.","labels":["city","country"],"threshold":0.3}' \
  | tee /tmp/gliner-token-e2e-extract.json
echo >&2
curl -sf "http://127.0.0.1:${GO_PORT}/v1/gliner" -H 'content-type: application/json' \
  -d '{"model":"gliner","text":"Kyiv is the capital of Ukraine.","labels":["city","country","river","person","car"],"threshold":0.3,"flat_ner":true,"model_type":"token","max_width":12,"max_length":768}' \
  | tee /tmp/gliner-token-e2e-gliner.json
echo >&2
python3 - <<'PY'
import json
ex=json.load(open("/tmp/gliner-token-e2e-extract.json"))
gl=json.load(open("/tmp/gliner-token-e2e-gliner.json"))
assert ex.get("entities"), ex
assert gl.get("entities"), gl
eng=gl.get("engine") or {}
assert eng.get("model_type") == "token", eng
texts=" ".join(e.get("text","") for e in gl["entities"]).lower()
assert "kyiv" in texts or "ukraine" in texts, gl
print("GLINER_TOKEN_GO_E2E_OK", "extract_n", len(ex["entities"]), "gliner_n", len(gl["entities"]))
PY
