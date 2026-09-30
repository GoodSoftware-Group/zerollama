#!/usr/bin/env bash
# GL5a: token-level multitask GLiNER ONNX smoke (GLiNER.cpp TOKEN_LEVEL).
# RelEx / bi-encoder remain parked — not in GLiNER.cpp yet (Finding 9).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PORT="${GLINER_PORT:-18096}"
MODEL_DIR="$(bash "${ROOT}/scripts/vendor/ensure_gliner_model.sh" multitask | tail -1)"
ONNX="${MODEL_DIR}/onnx/model.onnx"
TOK="${MODEL_DIR}/tokenizer.json"
BIN="${ZEROLLAMA_GLINER_SERVER_BIN:-}"

if [[ -z "${BIN}" ]]; then
  for c in "${ROOT}/gliner/server/build/gliner-server" "${ROOT}/gliner/server/build-cuda/gliner-server"; do
    [[ -x "${c}" ]] && BIN="${c}" && break
  done
fi
if [[ ! -x "${BIN:-}" ]]; then
  echo "skip: gliner-server not built" >&2
  exit 0
fi
case "${PORT}" in 11434|8081|8080) echo "error: reserved port" >&2; exit 1 ;; esac

# CPU ORT libs by default (multitask large is heavier — prefer CPU unless GLINER_ORT=cuda).
ORT_CPU="${ROOT}/vendor/onnxruntime-linux-x64-1.19.2"
export LD_LIBRARY_PATH="${ORT_CPU}/lib:/root/nvidia-host:${LD_LIBRARY_PATH:-}"
DEVICE_ARGS=()
if [[ "${GLINER_ORT:-cpu}" == "cuda" || "${GLINER_ORT:-}" == "gpu" ]]; then
  ORT_GPU="${ROOT}/vendor/onnxruntime-linux-x64-gpu-1.19.2"
  export LD_LIBRARY_PATH="${ORT_GPU}/lib:${LD_LIBRARY_PATH}"
  for d in /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib /usr/lib/ollama/mlx_cuda_v12; do
    [[ -d "${d}" ]] && export LD_LIBRARY_PATH="${d}:${LD_LIBRARY_PATH}"
  done
  DEVICE_ARGS+=(--device-id "${GLINER_DEVICE_ID:-0}")
fi

OWNED=0 PID=""
cleanup() { [[ "${OWNED}" == "1" && -n "${PID}" ]] && kill "${PID}" 2>/dev/null || true; }
trap cleanup EXIT

if ! curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  "${BIN}" --model "${ONNX}" --tokenizer "${TOK}" \
    --host 127.0.0.1 --port "${PORT}" --model-type token \
    --max-width 12 --max-length 768 \
    "${DEVICE_ARGS[@]}" \
    > /tmp/gliner-token-smoke.log 2>&1 &
  PID=$!; OWNED=1
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
    kill -0 "${PID}" 2>/dev/null || { tail -50 /tmp/gliner-token-smoke.log >&2; exit 1; }
    sleep 1
  done
fi

# Upstream example labels; Kyiv/Ukraine should hit after channel-first transpose (Finding 10 fix).
BODY='{"text":"Kyiv is the capital of Ukraine.","labels":["city","country","river","person","car"],"threshold":0.3,"model_type":"token","flat_ner":true,"max_width":12,"max_length":768}'
curl -sf "http://127.0.0.1:${PORT}/v1/gliner" -H 'content-type: application/json' -d "${BODY}" \
  | tee /tmp/gliner-token.json
echo >&2
python3 - <<'PY'
import json
doc=json.load(open("/tmp/gliner-token.json"))
assert "entities" in doc, doc
eng=doc.get("engine") or {}
assert eng.get("model_type") == "token", eng
ents=doc["entities"]
assert ents, f"expected token NER hits after logits transpose; got {doc}"
labels={e.get("label") for e in ents}
texts=" ".join(e.get("text","") for e in ents).lower()
assert "city" in labels or "country" in labels, ents
assert "kyiv" in texts or "ukraine" in texts, ents
print("GLINER_TOKEN_SMOKE_OK", "n", len(ents),
      [(e.get("text"), e.get("label"), round(float(e.get("score") or 0), 3)) for e in ents])
PY
