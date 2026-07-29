#!/usr/bin/env bash
# Minefield trap 12 — empty content at a thinking token ceiling.
#
# Lab only. Refuses production :11434 / :8081.
#
#   OLLAMA_HOST=127.0.0.1:11435 ./zerollama serve   # separately
#   ./scripts/minefield_ceiling_probe.sh qwen3:0.6b
#
# Env: BASE_URL (default http://127.0.0.1:11435), MAX_TOKENS (default 512)
set -euo pipefail

MODEL="${1:-}"
[[ -n "${MODEL}" ]] || { echo "usage: $0 <model-tag>" >&2; exit 2; }

BASE_URL="${BASE_URL:-http://127.0.0.1:11435}"
MAX_TOKENS="${MAX_TOKENS:-512}"
BASE_URL="${BASE_URL%/}"

case "${BASE_URL}" in
  *://127.0.0.1:11434*|*://localhost:11434*|*://127.0.0.1:8081*|*://localhost:8081*)
    echo "refusing production port in BASE_URL=${BASE_URL}" >&2
    exit 1
    ;;
esac

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

python3 - <<PY >"$TMP.req"
import json
print(json.dumps({
  "model": "${MODEL}",
  "stream": False,
  "think": True,
  "messages": [{"role": "user", "content":
    "Write a python function that validates RFC3339 timestamps without external libraries, with tests."}],
  "options": {"temperature": 0, "num_predict": int("${MAX_TOKENS}")},
}))
PY

curl -sS -m 300 "${BASE_URL}/api/chat" -H 'Content-Type: application/json' \
  -d @"$TMP.req" -o "$TMP"

python3 - "$TMP" <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
err = j.get("error")
if err:
    print("error:", err)
    sys.exit(2)
msg = j.get("message") or {}
content = (msg.get("content") or "").strip()
thinking = (msg.get("thinking") or "").strip()
done = j.get("done_reason") or j.get("finish_reason") or ""
print(f"content_chars={len(content)} thinking_chars={len(thinking)} done_reason={done!r}")
if not content and thinking:
    print("PROBLEM trap-12: empty content with non-empty thinking at this ceiling")
    sys.exit(1)
if not content and not thinking:
    print("PROBLEM: empty content and empty thinking")
    sys.exit(1)
print("CLEAN (or inconclusive): content present at this ceiling")
PY
