#!/usr/bin/env bash
# Lab smoke for DiffusionGemma DG3a marker slots (+ DG2c gather fallback).
# Never binds production ports. Free competing GPU jobs first (Finding 1).
#
# WHY assert non-zero logits: GPU sampling leaves host logits empty (Finding 7).
# WHY accept gather fallback: model may omit <<qid>> markers on short denoise.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${DIFFUSION_GGUF:-/root/models/diffusiongemma/diffusiongemma-26B-A4B-it-Q4_K_M.gguf}"
NGL="${DIFFUSION_NGL:-25}"
CTX="${DIFFUSION_CTX:-2048}"
PORT="${DIFFUSION_PORT:-18093}"
HOST="${DIFFUSION_HOST:-127.0.0.1}"
STEPS="${DIFFUSION_STEPS:-8}"

export LD_LIBRARY_PATH="/root/nvidia-host:${LD_LIBRARY_PATH:-}"

if [[ -z "${ZEROLLAMA_DIFFUSION_SERVER_BIN:-}" ]]; then
  CAND="${ROOT}/vendor/llama-cpp-diffusion-dd0cf0445/build/bin/llama-diffusion-gemma-server"
  if [[ -x "${CAND}" ]]; then
    ZEROLLAMA_DIFFUSION_SERVER_BIN="${CAND}"
  else
    echo "error: set ZEROLLAMA_DIFFUSION_SERVER_BIN or run ./scripts/build/build_llama_diffusion.sh" >&2
    exit 1
  fi
fi

if [[ ! -f "${MODEL}" ]]; then
  echo "error: missing ${MODEL}" >&2
  exit 1
fi

BASE="http://${HOST}:${PORT}"
OWNED_SERVER=0
TMPDIR_SMOKE="$(mktemp -d)"
cleanup() {
  if [[ "${OWNED_SERVER}" == "1" && -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
  fi
  rm -rf "${TMPDIR_SMOKE}"
}
trap cleanup EXIT

if ! curl -sf "${BASE}/props" >/dev/null 2>&1; then
  echo ">>> starting ${ZEROLLAMA_DIFFUSION_SERVER_BIN} -ngl ${NGL} -c ${CTX} :${PORT}" >&2
  "${ZEROLLAMA_DIFFUSION_SERVER_BIN}" \
    -m "${MODEL}" -ngl "${NGL}" -c "${CTX}" \
    --host "${HOST}" --port "${PORT}" > /tmp/dg3a-smoke.log 2>&1 &
  SERVER_PID=$!
  OWNED_SERVER=1
  for _ in $(seq 1 120); do
    if curl -sf "${BASE}/props" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "error: server exited; see /tmp/dg3a-smoke.log" >&2
      tail -40 /tmp/dg3a-smoke.log >&2 || true
      exit 1
    fi
    sleep 2
  done
fi

echo ">>> POST /tokenize" >&2
curl -sS "${BASE}/tokenize" -H 'content-type: application/json' -d '{"content":"billing"}' \
  > "${TMPDIR_SMOKE}/billing.json"
curl -sS "${BASE}/tokenize" -H 'content-type: application/json' -d '{"content":"tech"}' \
  > "${TMPDIR_SMOKE}/tech.json"
curl -sS "${BASE}/tokenize" -H 'content-type: application/json' -d '{"content":"<<dept>>"}' \
  > "${TMPDIR_SMOKE}/marker.json"
echo "billing=$(cat "${TMPDIR_SMOKE}/billing.json") tech=$(cat "${TMPDIR_SMOKE}/tech.json") marker=$(cat "${TMPDIR_SMOKE}/marker.json")" >&2

echo ">>> POST /v1/systemone (slots+gather)" >&2
python3 - "${BASE}" "${STEPS}" "${TMPDIR_SMOKE}" <<'PY'
import json, sys, urllib.request
base, steps, tmp = sys.argv[1], int(sys.argv[2]), sys.argv[3]
bill = json.load(open(f"{tmp}/billing.json"))
tech = json.load(open(f"{tmp}/tech.json"))
marker = json.load(open(f"{tmp}/marker.json"))
body = {
    "model": "openjev",
    "prompt": (
        'Reply with JSON {"dept":{"type":"choice","choice":"billing"}} '
        "then a marker line exactly:\n<<dept>> billing"
    ),
    "n_steps": steps,
    "max_tokens": 64,
    "readout": {
        "slots": [{
            "question_id": "dept",
            "marker_tokens": marker["tokens"],
            "options": [
                {"option_id": "billing", "tokens": bill["tokens"]},
                {"option_id": "tech", "tokens": tech["tokens"]},
            ],
        }],
        "gather": [
            {"question_id": "dept", "option_id": "billing", "tokens": bill["tokens"]},
            {"question_id": "dept", "option_id": "tech", "tokens": tech["tokens"]},
        ],
    },
}
req = urllib.request.Request(
    base + "/v1/systemone",
    data=json.dumps(body).encode(),
    headers={"content-type": "application/json"},
)
with urllib.request.urlopen(req, timeout=600) as r:
    raw = r.read().decode()
open(f"{tmp}/resp.json", "w").write(raw)
j = json.loads(raw)
g = j.get("gather") or []
mode = j.get("readout_mode")
assert mode in ("answer_marker_logit", "final_answer_logit_gather"), j
assert len(g) >= 2, g
vals = [float(h.get("logit_mean", 0)) for h in g]
assert any(abs(v) > 1e-6 for v in vals), f"all-zero logits (Finding 7 regression): {g}"
print(f"DG3a smoke OK mode={mode}:", {h["option_id"]: h["logit_mean"] for h in g})
print("answer_snip:", (j.get("answer") or "")[:180])
PY
