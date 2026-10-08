#!/usr/bin/env bash
# Lab smoke: tip llama-server --embeddings + score_fields through synthetic Clef GGUF.
#
# WHY synthetic: Cloudflare Clef is multi‑GB; public GGUF quants often omit the joint
# head. This proves tip b11351 wire (0135 + ZEROLLAMA_CLEF_DIR), not model quality.
#
# Ports: default 18086 (never 11434 / 8081 / production 8080).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PORT="${CLEF_SMOKE_PORT:-18086}"
SRC_GGUF="${CLEF_SMOKE_SRC:-/root/models/tiny-agent/Tiny-Agent-a-0.5B.F16.gguf}"
OUT_GGUF="${CLEF_SMOKE_GGUF:-/tmp/clef-synth-tiny.gguf}"
BIN="${CLEF_SMOKE_BIN:-${ROOT}/vendor/llama-cpp-b11351/build/bin/llama-server}"
NGL="${CLEF_SMOKE_NGL:-0}"

case "${PORT}" in
  11434|8080|8081) echo "error: refusing reserved port ${PORT}" >&2; exit 2 ;;
esac

if [[ ! -x "${BIN}" ]]; then
  echo "error: missing ${BIN}; build with CUDA_HOME=/usr/local/cuda-12.8 ./scripts/build/build_llama_server.sh" >&2
  exit 1
fi
if [[ ! -f "${SRC_GGUF}" ]]; then
  echo "error: missing source GGUF ${SRC_GGUF}" >&2
  exit 1
fi

if [[ "${CLEF_SMOKE_SKIP_GRAFT:-0}" != "1" || ! -f "${OUT_GGUF}" ]]; then
  python3 "${ROOT}/scripts/phase/l6_clef_graft_synth_gguf.py" -i "${SRC_GGUF}" -o "${OUT_GGUF}"
else
  echo "reusing ${OUT_GGUF} (CLEF_SMOKE_SKIP_GRAFT=1)"
fi

export LD_LIBRARY_PATH="$(dirname "${BIN}"):${LD_LIBRARY_PATH:-}"
fuser -k "${PORT}/tcp" 2>/dev/null || true
sleep 1

stdbuf -oL -eL "${BIN}" \
  -m "${OUT_GGUF}" \
  --host 127.0.0.1 --port "${PORT}" \
  -c 512 -ngl "${NGL}" --parallel 1 \
  --embeddings --pooling none \
  > /tmp/l6_clef_live_smoke_server.log 2>&1 &
SPID=$!
cleanup() { kill "${SPID}" 2>/dev/null || true; wait "${SPID}" 2>/dev/null || true; }
trap cleanup EXIT

READY=
for i in $(seq 1 120); do
  if ! kill -0 "${SPID}" 2>/dev/null; then
    echo "error: llama-server died; see /tmp/l6_clef_live_smoke_server.log" >&2
    tail -40 /tmp/l6_clef_live_smoke_server.log >&2
    exit 1
  fi
  code=$(curl -sS -o /tmp/l6_clef_tok.json -w '%{http_code}' --max-time 10 \
    "http://127.0.0.1:${PORT}/tokenize" \
    -H 'content-type: application/json' \
    -d '{"content":"STATE schema option yes option no"}' 2>/dev/null || echo 000)
  if [[ "${code}" == "200" ]]; then READY=$i; break; fi
  sleep 1
done
if [[ -z "${READY:-}" ]]; then
  echo "error: tokenize not ready" >&2
  tail -40 /tmp/l6_clef_live_smoke_server.log >&2
  exit 1
fi

TOKENS=$(python3 - <<'PY'
import json
d=json.load(open("/tmp/l6_clef_tok.json"))
# tip may return {"tokens":[...]} or list
if isinstance(d, list):
    print(",".join(str(t) for t in d))
elif "tokens" in d:
    print(",".join(str(t) for t in d["tokens"]))
else:
    raise SystemExit(d)
PY
)
N=$(python3 -c "print(len('${TOKENS}'.split(',')))")
if [[ "${N}" -lt 6 ]]; then
  echo "error: need >=6 tokens, got ${N}" >&2
  exit 1
fi

# token spans: question [0,2), options [2,4) and [4,6) — type 0 (noul)
python3 - <<PY
import json, urllib.request
tokens=[int(x) for x in "${TOKENS}".split(",")]
body={
  "input": tokens,
  "score_fields": [
    {"type": 0, "question": [0, 2], "options": [[2, 4], [4, 6]]}
  ],
}
req=urllib.request.Request(
  "http://127.0.0.1:${PORT}/embedding",
  data=json.dumps(body).encode(),
  headers={"content-type":"application/json"},
  method="POST",
)
with urllib.request.urlopen(req, timeout=180) as resp:
  out=json.load(resp)
print(json.dumps(out, indent=2)[:2000])
open("/tmp/l6_clef_live_smoke.json","w").write(json.dumps(out))
# expect [{"logits":[[...],[...]], "tokens_evaluated": N}]
assert isinstance(out, list) and len(out)==1, out
assert "logits" in out[0], out[0]
assert len(out[0]["logits"])==1 and len(out[0]["logits"][0])==2, out[0]
assert out[0].get("tokens_evaluated")==len(tokens), out[0]
print("PASS: l6_clef_live_smoke logits=", out[0]["logits"])
PY

# Go path: decision.Compile → /tokenize + /embedding score_fields → Answer
echo "== Go decision→score_fields =="
(
  cd "${ROOT}"
  CLEF_LIVE_PORT="${PORT}" /usr/local/go/bin/go run ./scripts/phase/l6_clef_go_score
)
echo "PASS: l6_clef_live_smoke (curl + Go)"
