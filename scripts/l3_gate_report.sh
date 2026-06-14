#!/usr/bin/env bash
# L3 gate verdict from l3_cache_smoke.json
#
# Usage: ./scripts/l3_gate_report.sh /tmp/l3-cache-smoke.json
set -euo pipefail

if [[ $# -lt 1 || ! -f "$1" ]]; then
  echo "usage: $0 <l3-cache-smoke.json>" >&2
  exit 1
fi

python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
print("== L3 gate report ==")
print(f"cache_key: {data.get('cache_key')}")
print(f"derived_slot: {data.get('derived_slot')}")
print(f"n_parallel: {data.get('n_parallel')}")
print(f"turn1_wall_s: {data.get('turn1_wall_s')}")
print(f"turn2_wall_s: {data.get('turn2_wall_s')}")
print(f"turn2_faster_than_turn1: {data.get('turn2_faster_than_turn1')}")
if data.get("speedup_ratio") is not None:
    print(f"speedup_ratio: {data.get('speedup_ratio')}")
if "cached_faster_than_no_cache" in data:
    print(f"cached_faster_than_no_cache: {data.get('cached_faster_than_no_cache')}")

lc = data.get("llama_cache") or {}
print(f"llama_cache.enabled: {lc.get('enabled')}")
print(f"slot_save_path: {lc.get('slot_save_path', '(n/a)')}")
if "disk_file_count" in data:
    print(f"disk_file_count: {data.get('disk_file_count')}")
    print(f"inprocess_disk_cache: {data.get('inprocess_disk_cache')}")

# Gate: strict = turn2 faster; soft = bridge wired (slot + cache enabled, both turns OK).
strict = bool(data.get("turn2_faster_than_turn1"))
soft = (
    data.get("derived_slot") is not None
    and (data.get("llama_cache") or {}).get("enabled") is True
    and data.get("turn1_wall_s") is not None
    and data.get("turn2_wall_s") is not None
)

print()
if strict or data.get("cached_faster_than_no_cache") is True:
    print("VERDICT: L3 cache smoke PASS (repeat turn latency improved)")
    sys.exit(0)
if soft:
    print("VERDICT: L3 cache smoke SOFT PASS (bridge wired; no latency win on this model/ctx)")
    print("  Try larger stable prefix or 27b @ 26k ctx for measurable prefill skip.")
    sys.exit(0)
print("VERDICT: L3 cache smoke FAIL — check subprocess backend + ZEROLLAMA_LLAMA_FORK=0 on stock")
sys.exit(1)
PY
