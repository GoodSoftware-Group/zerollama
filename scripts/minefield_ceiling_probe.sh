#!/usr/bin/env bash
# Minefield traps 12 + 22 — empty content at a thinking token ceiling.
#
# Trap 12: one budget can starve thinking and return empty content.
# Trap 22: the converting budget is per-model (and a distribution), not a
# family-card constant — run N>=3 and report conversion RATE, not a coin flip.
#
# Lab only. Refuses production :11434 / :8081.
#
#   OLLAMA_HOST=127.0.0.1:11435 ./zerollama serve   # separately
#   ./scripts/minefield_ceiling_probe.sh qwen3:0.6b
#   N=3 MAX_TOKENS=512 ./scripts/minefield_ceiling_probe.sh qwen3:0.6b
#
# Env: BASE_URL (default http://127.0.0.1:11435), MAX_TOKENS (default 512), N (default 3)
set -euo pipefail

MODEL="${1:-}"
[[ -n "${MODEL}" ]] || { echo "usage: $0 <model-tag>" >&2; exit 2; }

BASE_URL="${BASE_URL:-http://127.0.0.1:11435}"
MAX_TOKENS="${MAX_TOKENS:-512}"
N="${N:-3}"
BASE_URL="${BASE_URL%/}"

case "${BASE_URL}" in
  *://127.0.0.1:11434*|*://localhost:11434*|*://127.0.0.1:8081*|*://localhost:8081*)
    echo "refusing production port in BASE_URL=${BASE_URL}" >&2
    exit 1
    ;;
esac

if ! [[ "${N}" =~ ^[0-9]+$ ]] || [[ "${N}" -lt 1 ]]; then
  echo "N must be a positive integer (got ${N})" >&2
  exit 2
fi

echo "ceiling probe model=${MODEL} max_tokens=${MAX_TOKENS} n=${N} (trap 12/22)"

python3 - <<PY
import json, os, sys, urllib.request

base = os.environ.get("BASE_URL", "${BASE_URL}").rstrip("/")
model = "${MODEL}"
max_tokens = int("${MAX_TOKENS}")
n = int("${N}")

req_body = {
  "model": model,
  "stream": False,
  "think": True,
  "messages": [{"role": "user", "content":
    "Write a python function that validates RFC3339 timestamps without external libraries, with tests."}],
  "options": {"temperature": 0, "num_predict": max_tokens},
}

converted = 0
empty_think = 0
errors = 0
for i in range(n):
    data = json.dumps(req_body).encode()
    req = urllib.request.Request(base + "/api/chat", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            j = json.load(r)
    except Exception as e:
        print(f"sample {i+1}/{n}: ERROR {e}")
        errors += 1
        continue
    if j.get("error"):
        print(f"sample {i+1}/{n}: error {j['error']!r}")
        errors += 1
        continue
    msg = j.get("message") or {}
    content = (msg.get("content") or "").strip()
    thinking = (msg.get("thinking") or "").strip()
    done = j.get("done_reason") or j.get("finish_reason") or ""
    print(f"sample {i+1}/{n}: content_chars={len(content)} thinking_chars={len(thinking)} done_reason={done!r}")
    if content:
        converted += 1
    elif thinking:
        empty_think += 1

ok_n = n - errors
rate = (converted / ok_n) if ok_n else 0.0
print(f"conversion_rate={converted}/{ok_n} ({rate:.0%}) empty_content_with_thinking={empty_think} errors={errors}")
print("note: trap 22 — do not copy this ceiling to a sibling size; re-measure per model (n>=3)")
if ok_n == 0:
    sys.exit(2)
if empty_think and converted == 0:
    print("PROBLEM trap-12: all successful samples empty content with thinking at this ceiling")
    sys.exit(1)
if empty_think:
    print(f"PROBLEM trap-12/22: {empty_think}/{ok_n} empty-content-at-cap (distribution, not a yes/no)")
    sys.exit(1)
print("CLEAN: every successful sample produced content at this ceiling")
PY
