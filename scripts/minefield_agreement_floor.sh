#!/usr/bin/env bash
# Minefield trap 35 — tiny greedy agreement floor (same process, two passes).
#
# Not a full MMLU harness: runs a fixed item list twice at temperature=0 and
# reports per-item agreement. Lab only — refuses :11434 / :8081.
#
#   ./scripts/minefield_agreement_floor.sh qwen2.5:0.5b
#
# Env: BASE_URL, NUM_PREDICT (default 16)
set -euo pipefail

MODEL="${1:-}"
[[ -n "${MODEL}" ]] || { echo "usage: $0 <model-tag>" >&2; exit 2; }

BASE_URL="${BASE_URL:-http://127.0.0.1:11435}"
NUM_PREDICT="${NUM_PREDICT:-16}"
BASE_URL="${BASE_URL%/}"

case "${BASE_URL}" in
  *://127.0.0.1:11434*|*://localhost:11434*|*://127.0.0.1:8081*|*://localhost:8081*)
    echo "refusing production port in BASE_URL=${BASE_URL}" >&2
    exit 1
    ;;
esac

ITEMS="$(mktemp)"
PASS1="$(mktemp)"; PASS2="$(mktemp)"
trap 'rm -f "$ITEMS" "$PASS1" "$PASS2"' EXIT

cat >"$ITEMS" <<'EOF'
[
  "What is 2+2? Reply with only the number.",
  "Capital of France? One word.",
  "Is water wet? Yes or no.",
  "How many legs does a dog have? Number only.",
  "Color of the clear daytime sky? One word.",
  "What comes after Monday? One word.",
  "3*3=? Number only.",
  "Does the sun rise in the east? Yes or no."
]
EOF

run_pass() {
  python3 - "$BASE_URL" "$MODEL" "$NUM_PREDICT" "$ITEMS" <<'PY'
import json, sys, urllib.request
base, model, npred, items_path = sys.argv[1:5]
items = json.load(open(items_path))
rows = []
for q in items:
    body = json.dumps({
        "model": model,
        "stream": False,
        "prompt": q,
        "options": {"temperature": 0, "top_p": 1, "num_predict": int(npred)},
    }).encode()
    req = urllib.request.Request(base + "/api/generate", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        j = json.load(resp)
    rows.append({"q": q, "a": (j.get("response") or "").strip()})
print(json.dumps(rows))
PY
}

echo "agreement floor model=${MODEL} base=${BASE_URL} (same-process two passes)"
run_pass >"$PASS1"
run_pass >"$PASS2"

python3 - "$PASS1" "$PASS2" <<'PY'
import json, sys
a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
assert len(a) == len(b)
agree = sum(x["a"] == y["a"] for x, y in zip(a, b))
n = len(a)
print(f"item_agreement={agree}/{n} ({agree/n:.1%})")
for x, y in zip(a, b):
    if x["a"] != y["a"]:
        print(f"  [FLIP] {x['q']!r}")
        print(f"       pass1={x['a']!r}")
        print(f"       pass2={y['a']!r}")
print("Trap 35: treat disagreement as noise floor before publishing small deltas.")
print("Re-run after a server restart for the restart arm; use one host for paired arms.")
PY
