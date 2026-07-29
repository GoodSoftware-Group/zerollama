#!/usr/bin/env bash
# Minefield trap 54 — run order / warm-cache artifacts (temp-0 determinism + order reverse).
#
# Upstream: warm prefix / graph capture / run order can look like a real speedup or
# change temp=0 hashes. This lab probe reports:
#   1) unique response hashes across N identical temp=0 generates (apollo channel)
#   2) short↔long order reverse on prompt_eval_duration (second-wins = order artifact)
#
# Lab only — refuses :11434 / :8081.
#
#   ./scripts/minefield_warm_cache_check.sh qwen2.5:0.5b
#
# Env: BASE_URL (default http://127.0.0.1:11435), N (default 3), NUM_PREDICT (default 8)
set -euo pipefail

MODEL="${1:-}"
[[ -n "${MODEL}" ]] || { echo "usage: $0 <model-tag>" >&2; exit 2; }

BASE_URL="${BASE_URL:-http://127.0.0.1:11435}"
N="${N:-3}"
NUM_PREDICT="${NUM_PREDICT:-8}"
BASE_URL="${BASE_URL%/}"

case "${BASE_URL}" in
  *://127.0.0.1:11434*|*://localhost:11434*|*://127.0.0.1:8081*|*://localhost:8081*)
    echo "refusing production port in BASE_URL=${BASE_URL}" >&2
    exit 1
    ;;
esac

echo "warm-cache check model=${MODEL} base=${BASE_URL} N=${N} (trap 54)"

python3 - "$BASE_URL" "$MODEL" "$N" "$NUM_PREDICT" <<'PY'
import hashlib, json, sys, urllib.request

base, model, n_s, npred_s = sys.argv[1:5]
n, npred = int(n_s), int(npred_s)

def generate(prompt: str) -> dict:
    body = json.dumps({
        "model": model,
        "stream": False,
        "prompt": prompt,
        "options": {"temperature": 0, "top_p": 1, "num_predict": npred},
    }).encode()
    req = urllib.request.Request(
        base + "/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.load(resp)

# Shared inventory so later turns can reuse prefix KV / graph warm state.
words = " ".join(f"item{i:04d}" for i in range(120))
shared = (
    f"Inventory (unique tokens, do not repeat): {words}. "
    "Answer the next question with only the required token."
)
question = f"{shared} What is 2+2? Reply with only the number."

hashes = []
durs = []
for i in range(n):
    j = generate(question)
    text = (j.get("response") or "").strip()
    h = hashlib.sha256(text.encode()).hexdigest()[:12]
    pe = j.get("prompt_eval_duration") or 0
    hashes.append(h)
    durs.append(pe)
    print(f"  identical[{i}] hash={h} prompt_eval_ns={pe} reply={text[:40]!r}")

uniq = sorted(set(hashes))
print(f"unique_response_hashes={len(uniq)}/{n} hashes={uniq}")
if len(uniq) > 1:
    print("Trap 54 SIGNAL: temp=0 identical requests diverged (warm slot / prefix channel).")
else:
    print("Trap 54: temp=0 identical requests agreed (no hash drift this N).")

if len(durs) >= 2 and durs[0] and durs[-1]:
    ratio = durs[0] / max(durs[-1], 1)
    print(f"prompt_eval first/last ratio={ratio:.2f} (>>1 suggests warm-up paid on first)")

short = f"{shared} Reply with only: OK"
long = f"{shared} {' '.join(f'pad{i}' for i in range(80))} Reply with only: OK"

def pe(prompt: str) -> int:
    return int(generate(prompt).get("prompt_eval_duration") or 0)

print("order reverse: short→long then long→short")
ts1, tl1 = pe(short), pe(long)
tl2, ts2 = pe(long), pe(short)
print(f"  AB short_first={ts1} long_second={tl1}")
print(f"  BA long_first={tl2} short_second={ts2}")

def second_wins(first: int, second: int) -> bool:
    if first <= 0 or second <= 0:
        return False
    return second < first * 0.85  # ~15%+ faster when run second

short_second_win = second_wins(ts1, ts2)
long_second_win = second_wins(tl2, tl1)
if short_second_win and long_second_win:
    print("Trap 54 SIGNAL: whichever arm ran second was faster (order / warm-cache artifact).")
elif short_second_win or long_second_win:
    print("Trap 54 soft: one arm faster when second — counterbalance before publishing speedups.")
else:
    print("Trap 54: order reverse did not show a clean second-wins pattern.")

print("Protocol: reverse order, same-session baseline, null-build; never cross-session A vs B.")
print("Flush slot / restart between samples when hash drift appears; do not trust warm alone.")
PY
