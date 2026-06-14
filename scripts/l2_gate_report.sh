#!/usr/bin/env bash
# Print L2 exit-gate verdict from l2_metal_bench JSON (or multiple JSON paths).
#
# WHY: L2 merge decision needs a single readable summary — fork must win ≥2 of
# tok/s, max ctx, VRAM on measured legs without qwen35/runtime regressions.
#
# Usage:
#   ./scripts/l2_gate_report.sh /tmp/l2-metal-bench.json
#   ./scripts/l2_gate_report.sh /tmp/l2-metal-bench.json /tmp/l2-metal-bench-27k.json
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <l2-metal-bench.json> [...]" >&2
  exit 1
fi

python3 - "$@" <<'PY'
import json
import sys
from pathlib import Path

paths = [Path(p) for p in sys.argv[1:]]
reports = []
for p in paths:
    if not p.is_file():
        print(f"warn: missing {p}", file=sys.stderr)
        continue
    reports.append(json.loads(p.read_text()))

if not reports:
    print("no bench JSON found", file=sys.stderr)
    sys.exit(1)

print("== L2 gate report ==")
print(f"files: {', '.join(str(p) for p in paths if p.is_file())}")
print()

wins = {"decode": 0, "vram": 0, "max_ctx": 0}
legs_total = 0

for rep in reports:
    params = rep.get("bench_params") or {}
    num_ctx = params.get("num_ctx")
    comp = rep.get("comparison") or {}
    label = f"ctx={num_ctx}" if num_ctx else "unknown"
    print(f"--- {label} ---")
    if not comp:
        print("  (single-leg or no comparison)")
        for leg in (rep.get("legs") or {}).values():
            b = (leg.get("bench") or {})
            print(f"  {leg.get('label')}: {b.get('decode_tok_per_s')} tok/s")
        print()
        continue
    legs_total += 1
    s_tps = comp.get("stock_decode_tok_per_s")
    f_tps = comp.get("fork_decode_tok_per_s")
    d_win = comp.get("fork_wins_decode")
    v_win = comp.get("fork_wins_vram")
    print(f"  decode: stock={s_tps} fork={f_tps} delta={comp.get('decode_delta_pct')}% fork_wins={d_win}")
    print(f"  vram est: stock={comp.get('stock_vram_required_bytes')} fork={comp.get('fork_vram_required_bytes')} delta={comp.get('vram_delta_pct')}% fork_wins={v_win}")
    print(f"  cache: stock={comp.get('stock_cache')} fork={comp.get('fork_cache')}")
    smax = (rep.get("legs") or {}).get("stock", {}).get("vram_budget", {}).get("suggested_max_num_ctx")
    fmax = (rep.get("legs") or {}).get("fork", {}).get("vram_budget", {}).get("suggested_max_num_ctx")
    hw = (rep.get("legs") or {}).get("fork", {}).get("bench", {}).get("high_ctx_warmup_decodes")
    if hw:
        print(f"  high_ctx_warmup_decodes: {hw}")
    if smax is not None or fmax is not None:
        ctx_win = (fmax or 0) > (smax or 0)
        print(f"  suggested_max_num_ctx: stock={smax} fork={fmax} fork_wins={ctx_win}")
        if ctx_win:
            wins["max_ctx"] += 1
    if d_win:
        wins["decode"] += 1
    if v_win:
        wins["vram"] += 1
    print()

print("== aggregate (A/B legs only) ==")
print(f"  fork decode wins: {wins['decode']}/{legs_total}")
print(f"  fork vram wins:   {wins['vram']}/{legs_total}")
print(f"  fork max_ctx wins:{wins['max_ctx']}/{legs_total}")
score = sum(1 for k in ("decode", "vram", "max_ctx") if wins[k] > legs_total // 2 and legs_total)
# Gate: fork wins ≥2 of 3 dimensions across the benchmark suite (any leg counts per dimension).
dim_wins = sum(
    1
    for k, v in wins.items()
    if legs_total and v >= max(1, (legs_total + 1) // 2)
)
print()
if legs_total == 0:
    print("VERDICT: inconclusive (no stock vs fork comparisons)")
    sys.exit(2)
if dim_wins >= 2:
    print("VERDICT: fork PASSES partial gate on measured legs (≥2/3 dimensions)")
    print("  Still required: qwen35/runtime compat smokes + CUDA 5080 bench before vendor merge.")
    sys.exit(0)
print("VERDICT: fork FAILS merge gate on measured legs (stock wins most dimensions)")
print("  TBQ/QJL may still win at long ctx — run L2_NUM_CTX=26624 or 131072 fork-only leg.")
sys.exit(1)
PY
