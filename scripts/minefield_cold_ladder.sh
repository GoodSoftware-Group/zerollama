#!/usr/bin/env bash
# Minefield trap 61 — cold needle ladder (behavioural half).
# Also trap 60 — cold then warm on the same prompt (cache_reset vs reuse).
#
# Plants a fact at position 0, unique filler, decoy at the tail. Each depth:
#   1) cold (zerollama.cache_reset)
#   2) warm re-send byte-identical without reset
# Lab only — refuses :11434 / :8081.
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
  python3 - "$BASE_URL" "$MODEL" "$depth" "$NUM_PREDICT" "$FACT" "$DECOY" <<'PY'
import json, sys, urllib.request

base, model, depth_s, npred_s, fact, decoy = sys.argv[1:7]
depth, npred = int(depth_s), int(npred_s)
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

def run(label: str, cache_reset: bool) -> dict:
    opts = {"temperature": 0, "num_predict": npred}
    if cache_reset:
        opts["zerollama"] = {"cache_reset": True}
    body = json.dumps({
        "model": model,
        "stream": False,
        "prompt": prompt,
        "options": opts,
    }).encode()
    req = urllib.request.Request(
        base + "/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        j = json.load(resp)
    text = (j.get("response") or "").strip()
    row = {
        "label": label,
        "prompt_eval_count": j.get("prompt_eval_count"),
        "done_reason": j.get("done_reason") or "",
        "fact_recovered": fact in text,
        "decoy_leaked": decoy in text,
        "reply": text[:80],
        "prompt_eval_duration": j.get("prompt_eval_duration"),
    }
    print(
        f"depth≈{depth} {label} prompt_eval_count={row['prompt_eval_count']} "
        f"done_reason={row['done_reason']!r} fact_recovered={row['fact_recovered']} "
        f"decoy_leaked={row['decoy_leaked']} reply={row['reply']!r}"
    )
    return row

cold = run("cold", True)
warm = run("warm", False)
if cold["fact_recovered"] != warm["fact_recovered"] or cold["done_reason"] != warm["done_reason"]:
    print(
        f"depth≈{depth} Trap 60 SIGNAL: cold≠warm "
        f"fact {cold['fact_recovered']}→{warm['fact_recovered']} "
        f"done_reason {cold['done_reason']!r}→{warm['done_reason']!r}"
    )
elif cold["reply"] != warm["reply"]:
    print(f"depth≈{depth} Trap 60 soft: replies differ at temp=0 (hash/byte channel)")
else:
    print(f"depth≈{depth} Trap 60: cold/warm agreed on fact+done_reason this depth")
PY
done
echo "Note: depths are approximate char budgets, not tokenizer-exact. Compare prompt_eval_count."
echo "Trap 61: HTTP 200 + exact prompt tokens with fact_recovered=false is the silent-fail signature."
echo "Trap 60: report cold and warm as separate numbers; do not score only the warm retry."
