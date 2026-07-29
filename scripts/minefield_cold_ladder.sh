#!/usr/bin/env bash
# Minefield trap 61 — cold needle ladder (behavioural half).
#
# Plants a fact at position 0, unique filler, decoy at the tail. Runs cold
# (cache_reset) at a few token depths. Lab only — refuses :11434 / :8081.
#
#   ./scripts/minefield_cold_ladder.sh qwen2.5:0.5b
#
# Env: BASE_URL, DEPTHS (default "256,512,1024"), NUM_PREDICT (default 32)
set -euo pipefail

MODEL="${1:-}"
[[ -n "${MODEL}" ]] || { echo "usage: $0 <model-tag>" >&2; exit 2; }

BASE_URL="${BASE_URL:-http://127.0.0.1:11435}"
DEPTHS="${DEPTHS:-256,512,1024}"
NUM_PREDICT="${NUM_PREDICT:-32}"
FACT="${FACT:-NEON_PINEAPPLE_42}"
DECOY="${DECOY:-CRIMSON_BANANA_99}"
BASE_URL="${BASE_URL%/}"

case "${BASE_URL}" in
  *://127.0.0.1:11434*|*://localhost:11434*|*://127.0.0.1:8081*|*://localhost:8081*)
    echo "refusing production port in BASE_URL=${BASE_URL}" >&2
    exit 1
    ;;
esac

echo "cold ladder model=${MODEL} base=${BASE_URL} depths=${DEPTHS} fact=${FACT}"
IFS=',' read -r -a DEPTH_ARR <<<"${DEPTHS}"
for depth in "${DEPTH_ARR[@]}"; do
  depth="$(echo "$depth" | tr -d '[:space:]')"
  [[ -n "$depth" ]] || continue
  req="$(mktemp)"; resp="$(mktemp)"
  python3 - <<PY >"$req"
import json
fact, decoy, depth = "${FACT}", "${DECOY}", int("${depth}")
# Rough char budget: ~4 chars/token for filler words.
need = max(0, depth * 4 - 80)
words = []
i = 0
while len(" ".join(words)) < need:
    words.append(f"item{i:05d}")
    i += 1
filler = " ".join(words)
prompt = (
    f"The secret code is {fact}. It appears only once, at the start. "
    f"Inventory follows (unique lines): {filler}. "
    f"Decoy at the end: {decoy}. "
    f"What is the secret code? Reply with only the code."
)
print(json.dumps({
  "model": "${MODEL}",
  "stream": False,
  "prompt": prompt,
  "options": {
    "temperature": 0,
    "num_predict": int("${NUM_PREDICT}"),
    "zerollama": {"cache_reset": True},
  },
}))
PY
  code="$(curl -sS -m 300 -o "$resp" -w '%{http_code}' "${BASE_URL}/api/generate" \
    -H 'Content-Type: application/json' -d @"$req")"
  python3 - "$resp" "$code" "$depth" "$FACT" "$DECOY" <<'PY'
import json, sys
path, http, depth, fact, decoy = sys.argv[1:6]
raw = open(path).read()
try:
    j = json.loads(raw)
except Exception:
    print(f"depth={depth} http={http} parse_error body={raw[:200]!r}")
    sys.exit(0)
resp = (j.get("response") or "").strip()
eval_count = j.get("prompt_eval_count")
done = j.get("done_reason") or ""
hit = fact in resp
decoy_hit = decoy in resp
print(f"depth≈{depth} http={http} prompt_eval_count={eval_count} done_reason={done!r} "
      f"fact_recovered={hit} decoy_leaked={decoy_hit} reply={resp[:80]!r}")
PY
  rm -f "$req" "$resp"
done
echo "Note: depths are approximate char budgets, not tokenizer-exact. Compare prompt_eval_count."
echo "Trap 61: HTTP 200 + exact prompt tokens with fact_recovered=false is the silent-fail signature."
