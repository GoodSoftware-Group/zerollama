#!/usr/bin/env bash
# GL4: smoke gliner-server with CUDA ORT (--device-id 0). Lab ports only.
# Skips cleanly if no GPU, no CUDA binary, or no model.
#
# VRAM: do not co-reside with DiffusionGemma -ngl 25 / heavy emb on 16 GB —
# free those first (see docs/gliner-cpp-findings.md Finding 7).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PORT="${GLINER_PORT:-18095}"
MODEL_DIR="${GLINER_MODEL_DIR:-${HOME}/models/gliner_small-v2.1}"
ONNX="${GLINER_ONNX:-${MODEL_DIR}/onnx/model.onnx}"
TOK="${GLINER_TOKENIZER:-${MODEL_DIR}/tokenizer.json}"
DEVICE="${GLINER_DEVICE_ID:-0}"
BIN="${ZEROLLAMA_GLINER_SERVER_BIN:-}"

if [[ -z "${BIN}" ]]; then
  for c in \
    "${ROOT}/gliner/server/build-cuda/gliner-server" \
    "${ROOT}/gliner/server/build-cuda/bin/gliner-server" \
    "${ROOT}/gliner/server/build/gliner-server"
  do
    [[ -x "${c}" ]] && BIN="${c}" && break
  done
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "skip: no nvidia-smi (CPU-only host)" >&2
  exit 0
fi
if [[ ! -x "${BIN:-}" ]]; then
  echo "skip: gliner-server not built (GLINER_ORT=cuda ./scripts/build/build_gliner_server.sh)" >&2
  exit 0
fi
if [[ ! -f "${ONNX}" || ! -f "${TOK}" ]]; then
  echo "skip: model not found (set GLINER_MODEL_DIR)" >&2
  exit 0
fi
if [[ "${PORT}" == "11434" || "${PORT}" == "8081" || "${PORT}" == "8080" ]]; then
  echo "error: refusing production port ${PORT}" >&2
  exit 1
fi

# Prefer GPU ORT + cuDNN libs if present.
ORT_GPU="${ROOT}/vendor/onnxruntime-linux-x64-gpu-1.19.2"
export LD_LIBRARY_PATH="/root/nvidia-host:${LD_LIBRARY_PATH:-}"
if [[ -d "${ORT_GPU}/lib" ]]; then
  export LD_LIBRARY_PATH="${ORT_GPU}/lib:${LD_LIBRARY_PATH}"
fi
for d in \
  /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib \
  /usr/lib/ollama/mlx_cuda_v12
do
  [[ -d "${d}" ]] && export LD_LIBRARY_PATH="${d}:${LD_LIBRARY_PATH}"
done

OWNED=0 PID=""
cleanup() { [[ "${OWNED}" == "1" && -n "${PID}" ]] && kill "${PID}" 2>/dev/null || true; }
trap cleanup EXIT

echo ">>> nvidia-smi (pre)" >&2
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true

if ! curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  "${BIN}" --model "${ONNX}" --tokenizer "${TOK}" \
    --host 127.0.0.1 --port "${PORT}" --device-id "${DEVICE}" \
    > /tmp/gliner-cuda-smoke.log 2>&1 &
  PID=$!; OWNED=1
  for _ in $(seq 1 90); do
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
    if ! kill -0 "${PID}" 2>/dev/null; then
      echo "error: gliner-server exited; log:" >&2
      tail -60 /tmp/gliner-cuda-smoke.log >&2
      exit 1
    fi
    sleep 1
  done
fi

BODY='{"text":"Kyiv is the capital of Ukraine.","labels":["city","country"],"threshold":0.3,"device_id":'"${DEVICE}"',"model_type":"span"}'
curl -sf "http://127.0.0.1:${PORT}/v1/gliner" -H 'content-type: application/json' -d "${BODY}" \
  | tee /tmp/gliner-cuda.json
echo >&2

echo ">>> nvidia-smi (post)" >&2
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true

python3 - <<'PY'
import json
doc=json.load(open("/tmp/gliner-cuda.json"))
assert "entities" in doc and len(doc["entities"]) >= 1, doc
eng=doc.get("engine") or {}
assert eng.get("device_id", -1) >= 0, eng
print("GLINER_CUDA_SMOKE_OK", "n", len(doc["entities"]), "device_id", eng.get("device_id"))
PY
