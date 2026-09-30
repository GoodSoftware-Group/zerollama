#!/usr/bin/env bash
# Lab e2e: zerollama Go Decider → DiffusionGemma sibling (DG3a/b).
# Lab ports only — never 11434/8081. Free heavy GPU jobs first.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${DIFFUSION_GGUF:-/root/models/diffusiongemma/diffusiongemma-26B-A4B-it-Q4_K_M.gguf}"
NGL="${DIFFUSION_NGL:-8}"
CTX="${DIFFUSION_CTX:-2048}"
DIFF_PORT="${DIFFUSION_PORT:-18093}"
GO_PORT="${ZEROLLAMA_LAB_PORT:-11435}"

export LD_LIBRARY_PATH="/root/nvidia-host:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/go/bin:${PATH:-}"

if [[ -z "${ZEROLLAMA_DIFFUSION_SERVER_BIN:-}" ]]; then
  CAND="${ROOT}/vendor/llama-cpp-diffusion-dd0cf0445/build/bin/llama-diffusion-gemma-server"
  [[ -x "${CAND}" ]] || { echo "error: build llama-diffusion-gemma-server first" >&2; exit 1; }
  ZEROLLAMA_DIFFUSION_SERVER_BIN="${CAND}"
fi

ZL="${ROOT}/zerollama"
[[ -x "${ZL}" ]] || { echo "error: missing ${ZL}" >&2; exit 1; }
[[ -f "${MODEL}" ]] || { echo "error: missing ${MODEL}" >&2; exit 1; }

DIFF_PID=""
GO_PID=""
OWNED_DIFF=0
OWNED_GO=0
TMP="$(mktemp -d)"
cleanup() {
  [[ "${OWNED_GO}" == "1" && -n "${GO_PID}" ]] && kill "${GO_PID}" 2>/dev/null || true
  [[ "${OWNED_DIFF}" == "1" && -n "${DIFF_PID}" ]] && kill "${DIFF_PID}" 2>/dev/null || true
  rm -rf "${TMP}"
}
trap cleanup EXIT

if ! curl -sf "http://127.0.0.1:${DIFF_PORT}/props" >/dev/null 2>&1; then
  echo ">>> diffusion server :${DIFF_PORT} -ngl ${NGL}" >&2
  "${ZEROLLAMA_DIFFUSION_SERVER_BIN}" \
    -m "${MODEL}" -ngl "${NGL}" -c "${CTX}" \
    --host 127.0.0.1 --port "${DIFF_PORT}" > /tmp/dg-e2e-diff.log 2>&1 &
  DIFF_PID=$!
  OWNED_DIFF=1
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:${DIFF_PORT}/props" >/dev/null 2>&1 && break
    kill -0 "${DIFF_PID}" 2>/dev/null || { tail -40 /tmp/dg-e2e-diff.log >&2; exit 1; }
    sleep 2
  done
fi

if ! curl -sf "http://127.0.0.1:${GO_PORT}/api/version" >/dev/null 2>&1; then
  echo ">>> zerollama serve :${GO_PORT} OPENJEV→:${DIFF_PORT}" >&2
  OLLAMA_HOST="127.0.0.1:${GO_PORT}" \
  ZEROLLAMA_OPENJEV_URL="http://127.0.0.1:${DIFF_PORT}" \
  OLLAMA_NO_CLOUD=true \
  OLLAMA_TRAINING=false \
  ZEROLLAMA_RUNTIME_EMBED=0 \
    "${ZL}" serve > /tmp/dg-e2e-go.log 2>&1 &
  GO_PID=$!
  OWNED_GO=1
  for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:${GO_PORT}/api/version" >/dev/null 2>&1 && break
    kill -0 "${GO_PID}" 2>/dev/null || { tail -40 /tmp/dg-e2e-go.log >&2; exit 1; }
    sleep 1
  done
fi

echo ">>> POST /v1/systemone model=openjev" >&2
curl -sS "http://127.0.0.1:${GO_PORT}/v1/systemone" \
  -H 'content-type: application/json' \
  -d '{
    "model": "openjev",
    "state": {"body": "customer asking for refund on duplicate charge"},
    "questions": {
      "dept": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "refunds and charges", "tech": "bugs and outages"}
      }
    }
  }' > "${TMP}/resp.json"

python3 - "${TMP}/resp.json" <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
assert "answers" in j, j
ans = j["answers"]["dept"]
if isinstance(ans, str):
    ans = json.loads(ans)
assert ans.get("calibrated") is False, ans
assert "confidence" not in ans, ans
src = ans.get("score_source")
assert src in (
    "answer_marker_logit",
    "final_answer_logit_gather",
    "hybrid_marker_gather",
    None,
), ans
choice = ans.get("choice")
assert choice in ("billing", "tech") or ans.get("note"), ans
print("OpenJev e2e OK:", {"choice": choice, "score_source": src, "calibrated": ans.get("calibrated")})
PY
