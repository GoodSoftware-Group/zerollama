#!/usr/bin/env bash
# L3 gate verdict from l3_cache_smoke.json or merged l3_cuda_full_gate.json.
#
# Usage:
#   ./scripts/l3_gate_report.sh /tmp/l3-cache-smoke.json
#   ./scripts/l3_gate_report.sh /tmp/l3-cuda-full-gate/gate.json
set -euo pipefail

if [[ $# -lt 1 || ! -f "$1" ]]; then
  echo "usage: $0 <l3-smoke.json|l3-full-gate.json>" >&2
  exit 1
fi

python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path


def smoke_verdict(data):
    strict = bool(data.get("turn2_faster_than_turn1"))
    cached = data.get("cached_faster_than_no_cache") is True
    soft = (
        data.get("derived_slot") is not None
        and (data.get("llama_cache") or {}).get("enabled") is True
        and data.get("turn1_wall_s") is not None
        and data.get("turn2_wall_s") is not None
    )
    if strict or cached:
        return "pass", "latency improved (turn2 faster or cached vs no-cache)"
    if soft:
        return "soft", "bridge wired; no latency win on this model/ctx"
    return "fail", "check subprocess backend + ZEROLLAMA_LLAMA_CACHE=1"


def production_verdict(data):
    strict = bool(data.get("strict_pass"))
    cached = data.get("cached_faster_than_no_cache") is True
    faster = bool(data.get("turn2_faster_than_turn1"))
    ratio = data.get("turn2_over_turn1")
    if strict:
        return "pass", f"strict ratio {ratio} ≤ {data.get('strict_ratio_threshold', 0.75)}"
    if cached:
        return "pass", "cached faster than no-cache control"
    if faster:
        return "soft", f"turn2 faster than turn1 but ratio {ratio} above strict threshold"
    return "fail", "no cache latency win"


def leg_ran(data, name):
    legs = data.get("legs") or {}
    if name in legs:
        return bool(legs[name])
    # Legacy merged JSON: key present (even null) means orchestrator included the leg slot.
    if name in data:
        return data.get(name) is not None
    return False


def print_smoke(data, label=""):
    prefix = f"{label} " if label else ""
    print(f"== L3 {prefix}smoke ==")
    print(f"cache_key: {data.get('cache_key')}")
    print(f"derived_slot: {data.get('derived_slot')}")
    print(f"n_parallel: {data.get('n_parallel')}")
    print(f"num_ctx: {data.get('num_ctx')}")
    print(f"turn1_wall_s: {data.get('turn1_wall_s')}")
    print(f"turn2_wall_s: {data.get('turn2_wall_s')}")
    if data.get("turn2_no_cache_wall_s") is not None:
        print(f"turn2_no_cache_wall_s: {data.get('turn2_no_cache_wall_s')}")
    print(f"turn2_faster_than_turn1: {data.get('turn2_faster_than_turn1')}")
    if "cached_faster_than_no_cache" in data:
        print(f"cached_faster_than_no_cache: {data.get('cached_faster_than_no_cache')}")
    lc = data.get("llama_cache") or {}
    print(f"llama_cache.enabled: {lc.get('enabled')}")


def print_production(data):
    print("== L3 production (27k-class) ==")
    print(f"num_ctx: {data.get('num_ctx')}")
    print(f"prefix_repeat: {data.get('prefix_repeat')}")
    print(f"turn1_wall_s: {data.get('turn1_wall_s')}")
    print(f"turn2_wall_s: {data.get('turn2_wall_s')}")
    if data.get("no_cache_wall_s") is not None:
        print(f"no_cache_wall_s: {data.get('no_cache_wall_s')}")
    print(f"turn2_over_turn1: {data.get('turn2_over_turn1')}")
    print(f"strict_pass: {data.get('strict_pass')}")
    if "cached_faster_than_no_cache" in data:
        print(f"cached_faster_than_no_cache: {data.get('cached_faster_than_no_cache')}")


def main():
    data = json.loads(Path(sys.argv[1]).read_text())
    print("== L3 gate report ==")
    print(f"model: {data.get('model') or data.get('gguf', '(inline smoke)')}")

    if "smoke_8k" in data or "production_27k" in data:
        smoke = data.get("smoke_8k")
        prod = data.get("production_27k")
        ran_smoke = leg_ran(data, "smoke_8k")
        ran_prod = leg_ran(data, "production_27k")
        smoke_ok = smoke_ship = not ran_smoke
        prod_ok = prod_ship = not ran_prod

        if ran_smoke and smoke:
            print()
            print_smoke(smoke, "8k")
            sv, sm = smoke_verdict(smoke)
            print(f"8k verdict: {sv.upper()} — {sm}")
            smoke_ok = sv in ("pass", "soft")
            smoke_ship = sv == "pass"
        elif ran_smoke:
            print("\n8k smoke: (ran but no artifact)")
            smoke_ok = smoke_ship = False
        else:
            print("\n(skip: smoke_8k leg not run)")

        if ran_prod and prod:
            print()
            print_production(prod)
            pv, pm = production_verdict(prod)
            print(f"27k verdict: {pv.upper()} — {pm}")
            prod_ok = pv in ("pass", "soft")
            prod_ship = pv == "pass"
        elif ran_prod:
            print("\n27k production: (ran but no artifact)")
            prod_ok = prod_ship = False
        else:
            print("\n(skip: production_27k leg not run)")

        print()
        if ran_smoke and ran_prod:
            if smoke_ship and prod_ship:
                print("VERDICT: L3 CUDA full gate PASS (8k + 27k latency win)")
                return 0
            if not smoke_ok:
                print("VERDICT: L3 FAIL — 8k smoke leg failed")
            elif not prod_ok:
                print("VERDICT: L3 FAIL — 27k production leg failed")
            else:
                print("VERDICT: L3 FAIL — full gate requires cached-vs-no-cache win on both legs")
                if smoke_ok and not smoke_ship:
                    print("  note: 8k smoke SOFT only — use 7B+ with L3_PREFIX_REPEAT=150")
                if prod_ok and not prod_ship:
                    print("  note: 27k strict ratio not met and cached did not beat no-cache")
            return 1

        if ran_smoke and not ran_prod:
            if smoke_ship:
                print("VERDICT: L3 cache smoke PASS (8k only — production leg skipped)")
                return 0
            if smoke_ok:
                print("VERDICT: L3 cache smoke SOFT PASS (8k only — production leg skipped)")
                return 0
            print("VERDICT: L3 FAIL — 8k smoke leg failed")
            return 1

        if ran_prod and not ran_smoke:
            if prod_ship:
                print("VERDICT: L3 production gate PASS (27k only — smoke leg skipped)")
                return 0
            if prod_ok:
                print("VERDICT: L3 production gate SOFT PASS (27k only — smoke leg skipped)")
                return 0
            print("VERDICT: L3 FAIL — 27k production leg failed")
            return 1

        print("VERDICT: L3 FAIL — no gate legs ran")
        return 1

    print()
    print_smoke(data)
    sv, sm = smoke_verdict(data)
    print()
    if sv == "pass":
        print(f"VERDICT: L3 cache smoke PASS — {sm}")
        return 0
    if sv == "soft":
        print(f"VERDICT: L3 cache smoke SOFT PASS — {sm}")
        print("  Try L3_PREFIX_REPEAT=150+ on 7B+ or run l3_cuda_full_gate.sh on production GGUF.")
        return 0
    print(f"VERDICT: L3 cache smoke FAIL — {sm}")
    return 1


sys.exit(main())
PY
