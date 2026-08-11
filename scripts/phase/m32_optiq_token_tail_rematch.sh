#!/usr/bin/env bash
# Lab: F0700 in-process GRAPH token-tail on live MLX last-hidden rematch.
# Gate: serve generate with TOKEN_TAIL=require (GRAPH_GENERATE off) → suffix [12675,248046].
# Not F0699 cascade dylib — Go x/uma Buf/Graph APIs only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

UMA_TOOLKIT="${UMA_TOOLKIT:-}"
if [[ -z "$UMA_TOOLKIT" ]]; then
  for cand in \
    "$ROOT/../bmtl/hardware_lab/lanes/m4/uma_toolkit" \
    "$HOME/Sites/inference/bmtl/hardware_lab/lanes/m4/uma_toolkit"
  do
    if [[ -d "$cand" ]]; then
      UMA_TOOLKIT="$cand"
      break
    fi
  done
fi
if [[ -z "${UMA_TOOLKIT}" || ! -d "$UMA_TOOLKIT" ]]; then
  echo "FAIL: set UMA_TOOLKIT" >&2
  exit 1
fi
export UMA_TOOLKIT

M32_PORT="${M32_PORT:-11436}"
M32_URL="http://127.0.0.1:${M32_PORT}"
M32_MODEL="${M32_MODEL:-ornith-9b-optiq}"
M32_PROMPT="${M32_PROMPT:-Say hi in one word.}"
M32_NPRED="${M32_NPRED:-4}"
M32_NCTX="${M32_NCTX:-512}"
REF="${ORNITH_OPTIQ_TOKENS_REF:-/tmp/uma_mlxrunner_optiq_tokens_ref.json}"
DUMP_DIR="${UMA_OPTIQ_TOKEN_TAIL_DIR:-/tmp/uma_optiq_token_tail_dump}"
export UMA_OPTIQ_TOKEN_TAIL_DIR="$DUMP_DIR"
LOG_DIR="${M32_LOG_DIR:-/tmp/uma_m32_token_tail}"
mkdir -p "${LOG_DIR}"

if ! python3 - <<'PY'
import socket, sys
try:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(1)
    s.connect("/tmp/uma_daemon.sock")
    s.sendall(b"HELP\n")
    r = s.recv(4096).decode()
    s.close()
except Exception as e:
    print(f"FAIL: broker down ({e})", file=sys.stderr)
    sys.exit(1)
if "NORM_MUL_F16" not in r or "GEMV_Q8_G64" not in r:
    print(f"FAIL: broker lacks token-tail ops\n{r[:200]}", file=sys.stderr)
    sys.exit(1)
print("PASS: broker GRAPH ready")
PY
then
  exit 1
fi

echo "== ensure token-tail dump (fnorm + lm_q8) =="
make -C "$UMA_TOOLKIT" optiq-token-tail-dump
if [[ ! -f "$DUMP_DIR/lm_q8.bin" || ! -f "$DUMP_DIR/meta.txt" ]]; then
  echo "FAIL: missing $DUMP_DIR/lm_q8.bin (blobs?)" >&2
  exit 1
fi
echo "PASS: dump at $DUMP_DIR"

if [[ ! -f "$REF" ]]; then
  echo "FAIL: missing $REF (run m26 freeze first)" >&2
  exit 1
fi

echo "== Go OptiqTokenTailArgmax smoke (post-norm dump / gemv_argmax) =="
make -C x/uma
export ZEROLLAMA_UMA_OPTIQ_TOKEN_SMOKE=1
CGO_ENABLED=1 go test -tags uma -count=1 -timeout 30m ./x/uma/ -run TestOptiqTokenTailPrefillRematch

_pick_bin() {
  if [[ -x "${ROOT}/zerollama" ]]; then echo "${ROOT}/zerollama"; return; fi
  if command -v zerollama >/dev/null 2>&1; then command -v zerollama; return; fi
  echo ""
}

echo "== build zerollama -tags uma =="
CGO_ENABLED=1 go build -tags uma -o zerollama .
BIN="$(_pick_bin)"
if [[ -z "${BIN}" ]]; then
  echo "FAIL: no zerollama binary" >&2
  exit 1
fi

_stop() {
  if [[ -f "${LOG_DIR}/serve.pid" ]]; then
    kill "$(cat "${LOG_DIR}/serve.pid")" 2>/dev/null || true
    rm -f "${LOG_DIR}/serve.pid"
    sleep 0.5
  fi
}
trap '_stop' EXIT INT TERM

_wait_ready() {
  for _ in $(seq 1 180); do
    if curl -sS --max-time 10 "${M32_URL}/api/tags" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  echo "FAIL: serve not ready on ${M32_URL}" >&2
  tail -80 "${LOG_DIR}/serve.log" 2>/dev/null || true
  exit 1
}

echo "== serve TOKEN_TAIL=require GRAPH_GENERATE=off =="
_stop
# Critical: do NOT set GRAPH_GENERATE — that is F0699 cascade path.
unset ZEROLLAMA_UMA_OPTIQ_GRAPH_GENERATE || true
OLLAMA_HOST="127.0.0.1:${M32_PORT}" \
  ZEROLLAMA_UMA_SCHED=require \
  ZEROLLAMA_UMA_OPTIQ_TOKEN_TAIL=require \
  ZEROLLAMA_UMA_SCHED_LOG=1 \
  ZEROLLAMA_RUNTIME_DARWIN_SIDECAR=0 \
  UMA_OPTIQ_TOKEN_TAIL_DIR="$DUMP_DIR" \
  OLLAMA_TRAINING=false \
  "${BIN}" serve >"${LOG_DIR}/serve.log" 2>&1 &
echo $! >"${LOG_DIR}/serve.pid"
_wait_ready

GEN_OUT="${LOG_DIR}/generate.json"
python3 - "${M32_MODEL}" "${M32_PROMPT}" "${M32_NPRED}" "${M32_NCTX}" "${M32_URL}" "${GEN_OUT}" <<'PY'
import json, sys, urllib.request
model, prompt, npred, nctx, url, out = sys.argv[1:7]
req = {
    "model": model,
    "prompt": prompt,
    "stream": False,
    "options": {
        "num_predict": int(npred),
        "num_ctx": int(nctx),
        "temperature": 0,
        "repeat_penalty": 1.0,
        "repeat_last_n": 0,
    },
}
data = json.dumps(req).encode()
r = urllib.request.urlopen(urllib.request.Request(url + "/api/generate", data=data,
                                                  headers={"Content-Type": "application/json"}),
                           timeout=900)
body = r.read()
open(out, "wb").write(body)
d = json.loads(body)
if "error" in d:
    raise SystemExit("generate error: " + str(d["error"]))
if d.get("context") is None:
    raise SystemExit("missing context in response")
print("thinking:", (d.get("thinking") or "")[:40])
print("n_ctx_tokens:", len(d["context"]))
print("context_suffix:", d["context"][-4:])
PY

python3 - "${GEN_OUT}" "${REF}" <<'PY'
import json, sys
got = json.load(open(sys.argv[1]))
ref = json.load(open(sys.argv[2]))
ctx = got["context"]
want = [12675, 248046]
# Prefer tokens_ref generate suffix via prompt_eval_count.
pec = ref.get("prompt_eval_count") or 0
if pec > 0 and pec < len(ref.get("tokens") or []):
    want = list(ref["tokens"][pec:])
got_suf = ctx[-len(want):]
print("got_suffix:", got_suf)
print("want_suffix:", want)
if got_suf != want:
    raise SystemExit(f"FAIL: suffix rematch got={got_suf} want={want}")
print("PASS: TOKEN_TAIL generate suffix rematch", want)
PY

echo "== pipeline hooks present =="
rg -n 'OptiqTokenTail|beginOptiqTokenTail|maybeOwnedGraphToken|SkipFinalNorm|TOKEN_TAIL' \
  x/mlxrunner/pipeline.go x/mlxrunner/optiq_token_tail.go x/uma/optiq_token_tail.go \
  x/models/qwen3_5/qwen3_5.go
echo "PASS: F0700 hooks wired"

echo "m32 optiq token-tail rematch PASS"
