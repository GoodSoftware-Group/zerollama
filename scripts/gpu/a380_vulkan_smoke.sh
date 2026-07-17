#!/usr/bin/env bash
# Vulkan API smoke for Intel Arc A380 — measures load_ms + total_duration_eval_tok_s.
#
# WHY: eval_tok_s alone (~43) hides ~580ms load_duration per request (research lane).
# Uses asm_lab benchmark when present; falls back to curl + jq.
#
#   source ./scripts/gpu/a380_env.sh
#   ./scripts/gpu/a380_vulkan_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/gpu/a380_env.sh
source "${ROOT}/scripts/gpu/a380_env.sh"

OUT="${A380_SMOKE_OUT:-/tmp/a380-vulkan-smoke.json}"
HOST="${OLLAMA_HOST#http://}"
HOST="${HOST#https://}"
API="http://${HOST}"

MODEL="${A380_SIGNOFF_MODEL}"
PROMPT="${A380_SIGNOFF_PROMPT}"
TOKENS="${A380_SIGNOFF_TOKENS}"

BENCH="${ZA380_RESEARCH_LANE}/scripts/ollama_benchmark.py"
if [[ "${A380_ZEROLLAMA_SMOKE:-1}" == "1" ]]; then
  echo "== a380 vulkan smoke (zerollama API) =="
  python3 - <<PY
import json, os, time, urllib.request
from pathlib import Path

host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
if not host.startswith("http"):
    host = f"http://{host}"
model = os.environ["A380_SIGNOFF_MODEL"]
prompt = os.environ.get("A380_SIGNOFF_PROMPT", "hello")
tokens = int(os.environ.get("A380_SIGNOFF_TOKENS", "8"))
out = Path("${OUT}")

def bench(label, keep_alive=None):
    payload = {"model": model, "prompt": prompt, "stream": False,
               "options": {"num_predict": tokens}}
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=data,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        r = json.loads(resp.read())
    wall = time.perf_counter() - t0
    ec = r.get("eval_count") or 0
    ed = r.get("eval_duration") or 1
    td = r.get("total_duration") or 1
    ld = r.get("load_duration") or 0
    return {
        "label": label, "status": "measured", "eval_count": ec,
        "load_ms": round(ld / 1e6, 2),
        "eval_tok_s": round(ec / (ed / 1e9), 3) if ed else None,
        "total_duration_eval_tok_s": round(ec / (td / 1e9), 3) if td and ec else None,
        "wall_eval_tok_s": round(ec / wall, 3) if wall and ec else None,
        "wall_s": round(wall, 3),
    }

rows = [bench("zerollama_vulkan_cold")]
for i in range(1, 4):
    rows.append(bench(f"zerollama_vulkan_warm_{i}"))
warm = [r for r in rows if r.get("eval_tok_s")]
best = max(warm, key=lambda r: r["eval_tok_s"] or 0)
doc = {
    "target": "arc-a380",
    "probe_id": "zerollama-vulkan-smoke",
    "rows": rows,
    "interpretation": {
        "backend": "zerollama_vulkan",
        "ollama_host": host,
        "best_warm_eval_tok_s": best.get("eval_tok_s"),
        "warm_load_ms_avg": round(sum(r["load_ms"] for r in rows[1:]) / 3, 1),
    },
}
out.write_text(json.dumps(doc, indent=2) + "\n")
print(f"wrote {out}")
for r in rows:
    print(f"{r['label']:28} eval={r.get('eval_tok_s')} total={r.get('total_duration_eval_tok_s')} load_ms={r.get('load_ms')}")
PY
elif [[ -f "${BENCH}" ]]; then
  echo "== a380 vulkan smoke (asm_lab benchmark) =="
  python3 "${BENCH}" \
    --out "${OUT}" \
    --ollama-model "${MODEL}" \
    --gguf "${A380_SIGNOFF_GGUF}" \
    --prompt "${PROMPT}" \
    --tokens "${TOKENS}"
else
  echo "== a380 vulkan smoke (curl fallback) =="
  RESP="$(curl -sf -m 120 "${API}/api/generate" -d "$(jq -nc \
    --arg m "$MODEL" --arg p "$PROMPT" --argjson n "$TOKENS" \
    '{model:$m, prompt:$p, stream:false, options:{num_predict:$n}}')")"
  echo "$RESP" | jq '{
    load_ms: (.load_duration/1e6),
    eval_tok_s: (.eval_count / (.eval_duration/1e9)),
    total_duration_eval_tok_s: (.eval_count / (.total_duration/1e9)),
    eval_count, total_duration, load_duration, eval_duration
  }' | tee "${OUT}"
fi

echo "== thresholds (soft) =="
python3 - <<PY
import json, os, sys
from pathlib import Path
p = Path("${OUT}")
raw = json.loads(p.read_text())
# asm_lab benchmark wraps rows; curl fallback is flat
rows = raw.get("rows") if isinstance(raw.get("rows"), list) else [raw]
warm = [r for r in rows if r.get("eval_count")]
if not warm:
    print("no measured rows", file=sys.stderr); sys.exit(1)
r = warm[-1]
load_ms = float(r.get("load_ms") or (r.get("load_duration", 0) / 1e6))
eval_tok = float(r.get("eval_tok_s") or 0)
total_tok = float(r.get("total_duration_eval_tok_s") or r.get("wall_eval_tok_s") or 0)
min_eval = float(os.environ.get("A380_MIN_EVAL_TOK_S", "38"))
min_total = float(os.environ.get("A380_MIN_TOTAL_TOK_S_8", "8"))
max_load = float(os.environ.get("A380_MAX_LOAD_MS", "700"))
ok = True
if eval_tok and eval_tok < min_eval:
    print(f"WARN eval_tok_s {eval_tok:.1f} < {min_eval}")
    ok = False
if total_tok and total_tok < min_total:
    print(f"WARN total_duration_eval_tok_s {total_tok:.1f} < {min_total}")
    ok = False
if load_ms > max_load:
    print(f"WARN load_ms {load_ms:.0f} > {max_load}")
print(f"load_ms={load_ms:.0f} eval_tok_s={eval_tok:.1f} total_duration_eval_tok_s={total_tok:.1f}")
sys.exit(0 if ok else 2)
PY
