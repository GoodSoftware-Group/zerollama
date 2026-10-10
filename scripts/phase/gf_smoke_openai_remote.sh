#!/usr/bin/env bash
# GF3 lab smoke — openai-remote tag → glm-flash-lite sidecar.
# Never binds production :11434 / :8081. Default lab host :11436.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SIDECAR="${GF_SIDECAR:-http://127.0.0.1:30000}"
# Strip scheme if operator exported OLLAMA_HOST=http://… (common); force lab port.
_raw_host="${GF_OLLAMA_HOST:-127.0.0.1:11436}"
HOST="${_raw_host#http://}"
HOST="${HOST#https://}"
HOST="${HOST%/}"
BIN="${ZEROLLAMA_BIN:-$ROOT/zerollama}"
MODELS="${OLLAMA_MODELS:-${MODELS:-}}"

echo "=== sidecar health $SIDECAR ==="
curl -sS --max-time 5 "$SIDECAR/health" | tee /tmp/gf-smoke-health.json
echo
curl -sS --max-time 5 "$SIDECAR/v1/models" | head -c 200
echo

if [[ ! -x "$BIN" ]]; then
  echo "build $BIN"
  (cd "$ROOT" && CGO_ENABLED=1 go build -o "$BIN" .)
fi

echo "=== register tag (idempotent) ==="
OLLAMA_MODELS="${MODELS:-}" "$ROOT/scripts/phase/gf3_register_openai_remote.sh"

# If nothing listens on HOST, start a short-lived lab serve.
BASE="http://${HOST}"
if ! curl -sS --max-time 1 "$BASE/api/version" >/dev/null 2>&1; then
  echo "=== start lab serve $HOST ==="
  env OLLAMA_HOST="$HOST" \
      ${MODELS:+OLLAMA_MODELS="$MODELS"} \
      ZEROLLAMA_RUNTIME_EMBED=0 \
      OLLAMA_TRAINING=false \
      OLLAMA_NO_CLOUD=true \
      "$BIN" serve > /tmp/gf-smoke-serve.log 2>&1 &
  SPID=$!
  trap 'kill '"$SPID"' 2>/dev/null || true' EXIT
  for i in $(seq 1 60); do
    curl -sS --max-time 1 "$BASE/api/version" >/dev/null 2>&1 && break
    sleep 0.5
  done
fi

echo "=== /v1/chat/completions ==="
curl -sS --max-time 180 "$BASE/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Say hi in four words."}],"max_tokens":24,"chat_template_kwargs":{"enable_thinking":false}}' \
  | tee /tmp/gf-smoke-v1.json
echo

echo "=== /api/chat ==="
curl -sS --max-time 180 "$BASE/api/chat" \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Say hi in four words."}],"stream":false,"options":{"num_predict":24}}' \
  | tee /tmp/gf-smoke-api.json
echo

python3 - <<'PY'
import json
for path in ("/tmp/gf-smoke-v1.json", "/tmp/gf-smoke-api.json"):
    r = json.load(open(path))
    if "error" in r and not r.get("choices") and not r.get("message"):
        raise SystemExit(f"FAIL {path}: {r}")
    if "choices" in r:
        c = (r["choices"][0].get("message") or {}).get("content", "")
        print(path, "finish", r["choices"][0].get("finish_reason"), "→", c[:80])
    else:
        print(path, "done", r.get("done"), "→", (r.get("message") or {}).get("content", "")[:80])
print("GF smoke PASS")
PY
