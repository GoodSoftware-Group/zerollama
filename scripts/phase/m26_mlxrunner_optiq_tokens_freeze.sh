#!/usr/bin/env bash
# Lab: live mlxrunner OptiQ tokens_ref freeze (wishlist GRAPH-MLX 4.6 / F0626+F0686).
# Freezes greedy *sampled* context token ids from live mlxrunner, rematches require + off.
# F0686: serve Context prefers mlxrunner Done.Tokens (sampled), not re-tokenized text.
# Options: temperature=0, repeat_penalty=1.0, repeat_last_n=0 so greedy matches raw
# Forward argmax (default repeat_penalty=1.1 diverted away from <|im_end|> in prompt).
# Generate stops at EOS (<|im_end|>) — may be shorter than num_predict.
# Never touches :11434 / :8081.
#
#   M26_MODEL=ornith-9b-optiq ./scripts/phase/m26_mlxrunner_optiq_tokens_freeze.sh
#   M26_MODEL=gemma4:26b-optiq …   # heavier
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

M26_PORT="${M26_PORT:-11435}"
M26_URL="http://127.0.0.1:${M26_PORT}"
M26_MODEL="${M26_MODEL:-ornith-9b-optiq}"
M26_PROMPT="${M26_PROMPT:-Say hi in one word.}"
M26_NPRED="${M26_NPRED:-4}"
M26_NCTX="${M26_NCTX:-512}"
REF="${M26_TOKENS_REF:-/tmp/uma_mlxrunner_optiq_tokens_ref.json}"
LOG_DIR="${M26_LOG_DIR:-/tmp/uma_m26_freeze}"
mkdir -p "${LOG_DIR}"

_pick_bin() {
  if [[ -x "${ROOT}/zerollama" ]]; then echo "${ROOT}/zerollama"; return; fi
  if command -v zerollama >/dev/null 2>&1; then command -v zerollama; return; fi
  echo ""
}

BIN="$(_pick_bin)"
if [[ -z "${BIN}" ]]; then
  echo "FAIL: no zerollama binary" >&2
  exit 1
fi

echo "== F0626 live mlxrunner OptiQ tokens_ref freeze =="
echo "bin=${BIN} port=${M26_PORT} model=${M26_MODEL} ref=${REF}"

# Broker required for require-mode rematch.
if ! /Users/user1/Sites/inference/bmtl/hardware_lab/lanes/m4/uma_toolkit/uma_daemon --ping >/dev/null 2>&1; then
  echo "== start uma_daemon =="
  (cd /Users/user1/Sites/inference/bmtl/hardware_lab/lanes/m4/uma_toolkit && \
    pkill -x uma_daemon 2>/dev/null || true
    sleep 0.3
    rm -f /tmp/uma_daemon.sock
    nohup ./uma_daemon >/tmp/uma_m26_daemon.log 2>&1 &
    sleep 2
    ./uma_daemon --ping)
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
  # /api/tags can take >2s when the process is busy; allow 10s per probe.
  for _ in $(seq 1 180); do
    if curl -sS --max-time 10 "${M26_URL}/api/tags" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  echo "FAIL: serve not ready on ${M26_URL}" >&2
  tail -40 "${LOG_DIR}"/serve-*.log 2>/dev/null || true
  exit 1
}

_start() {
  local mode="$1"
  _stop
  local log="${LOG_DIR}/serve-${mode}.log"
  echo "== serve UMA=${mode} =="
  OLLAMA_HOST="127.0.0.1:${M26_PORT}" \
    ZEROLLAMA_UMA_SCHED="${mode}" \
    ZEROLLAMA_UMA_SCHED_LOG=1 \
    ZEROLLAMA_RUNTIME_DARWIN_SIDECAR=0 \
    OLLAMA_TRAINING=false \
    "${BIN}" serve >"${log}" 2>&1 &
  echo $! >"${LOG_DIR}/serve.pid"
  _wait_ready
}

_generate() {
  local out="$1"
  python3 - "${M26_MODEL}" "${M26_PROMPT}" "${M26_NPRED}" "${M26_NCTX}" "${M26_URL}" "${out}" <<'PY'
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
        # F0686: disable default repeat_penalty (1.1). Prompt already contains
        # <|im_end|>; penalty diverted greedy away from raw argmax (lab got_gen).
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
print("response:", (d.get("response") or "").replace("\n", " ")[:80])
print("n_ctx_tokens:", len(d["context"]))
PY
}

_write_ref() {
  local gen_json="$1"
  python3 - "${gen_json}" "${REF}" "${M26_MODEL}" "${M26_PROMPT}" <<'PY'
import json, sys
gen, ref, model, prompt = sys.argv[1:5]
d = json.load(open(gen))
toks = d["context"]
obj = {
    "model": model,
    "prompt": prompt,
    "temperature": 0,
    "repeat_penalty": 1.0,
    "repeat_last_n": 0,
    "ngen": len(toks),
    "tokens": toks,
    "response": d.get("response", ""),
    "thinking": d.get("thinking", ""),
    "eval_count": d.get("eval_count"),
    "prompt_eval_count": d.get("prompt_eval_count"),
    "done_reason": d.get("done_reason", ""),
}
json.dump(obj, open(ref, "w"), indent=2)
open(ref, "a").write("\n")
print("wrote", ref, "ngen=", len(toks))
PY
}

_rematch() {
  local gen_json="$1"
  local label="$2"
  python3 - "${gen_json}" "${REF}" "${label}" <<'PY'
import json, sys
gen, ref, label = sys.argv[1:4]
g = json.load(open(gen))
r = json.load(open(ref))
got, want = g["context"], r["tokens"]
if got != want:
    print(f"FAIL: {label} tokens != freeze", file=sys.stderr)
    print("want[:16]=", want[:16], file=sys.stderr)
    print("got[:16]=", got[:16], file=sys.stderr)
    # find first mismatch
    for i, (a, b) in enumerate(zip(want, got)):
        if a != b:
            print(f"mismatch @{i}: want={a} got={b}", file=sys.stderr)
            break
    if len(want) != len(got):
        print(f"len want={len(want)} got={len(got)}", file=sys.stderr)
    sys.exit(1)
print(f"PASS: {label} matches tokens_ref (n={len(want)})")
PY
}

# Skip cleanly if model absent.
if ! "${BIN}" show "${M26_MODEL}" >/dev/null 2>&1; then
  echo "SKIP: model ${M26_MODEL} not present (ollama show failed)" >&2
  exit 0
fi

_start require
_generate "${LOG_DIR}/gen-freeze.json"
_write_ref "${LOG_DIR}/gen-freeze.json"
_generate "${LOG_DIR}/gen-require-rematch.json"
_rematch "${LOG_DIR}/gen-require-rematch.json" "require-rematch"

_start off
_generate "${LOG_DIR}/gen-off-rematch.json"
_rematch "${LOG_DIR}/gen-off-rematch.json" "off-rematch"

echo "F0626 live mlxrunner OptiQ tokens_ref freeze PASS"
echo "  ref=${REF}"
