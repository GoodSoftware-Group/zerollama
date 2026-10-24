#!/usr/bin/env python3
"""A/B two (or more) routers on a labeled prompt set via POST /api/router/decide.

AR4 exit check (docs/auto-router.md): compare `classifier: jev` against `classifier: score`
on the same policies/candidates. Dry runs only — nothing generates, so it is safe on a lab
node. Stdlib only.

    ./scripts/eval/router_ab.py --host http://127.0.0.1:11435 \
        --routers agent-score agent-jev scripts/eval/router_ab_sample.jsonl

Input JSONL rows: {"prompt": "...", "expected": "<model tag>"}  ("expected" optional).

WHY lab port default: never probe production :11434 from agent work (AGENTS.md).
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def decide(host, router, prompt, timeout):
    body = json.dumps({"router": router, "prompt": prompt}).encode()
    req = urllib.request.Request(
        host.rstrip("/") + "/api/router/decide",
        data=body,
        headers={"content-type": "application/json"},
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.load(resp)
            out["_ms"] = (time.monotonic() - start) * 1000
            return out
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}",
                "_ms": (time.monotonic() - start) * 1000}
    except (urllib.error.URLError, TimeoutError) as e:
        return {"error": str(e), "_ms": (time.monotonic() - start) * 1000}


def same_model(a, b):
    """Compare tags loosely: 'qwen3' == 'qwen3:latest'."""
    norm = lambda m: m if ":" in m else m + ":latest"
    return bool(a) and bool(b) and norm(a) == norm(b)


def pct(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompts", help="JSONL with prompt / expected")
    ap.add_argument("--routers", nargs="+", required=True, help="router names from router.yaml")
    ap.add_argument("--host", default="http://127.0.0.1:11435")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--verbose", action="store_true", help="print every disagreement")
    args = ap.parse_args()

    rows = []
    with open(args.prompts, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(json.loads(line))
    if not rows:
        sys.exit("no prompts")

    results = {r: [] for r in args.routers}
    for row in rows:
        for r in args.routers:
            results[r].append(decide(args.host, r, row["prompt"], args.timeout))

    labeled = [i for i, row in enumerate(rows) if row.get("expected")]
    print(f"{len(rows)} prompts ({len(labeled)} labeled) against {args.host}\n")
    print(f"{'router':<24}{'accuracy':>10}{'errors':>8}{'fallback':>10}{'p50 ms':>9}{'p95 ms':>9}")
    for r in args.routers:
        res = results[r]
        errs = sum(1 for x in res if "error" in x)
        fb = sum(1 for x in res if x.get("fallback"))
        ms = [x["_ms"] for x in res]
        acc = "-"
        if labeled:
            hits = sum(1 for i in labeled if same_model(res[i].get("candidate"), rows[i]["expected"]))
            acc = f"{100 * hits / len(labeled):.0f}%"
        print(f"{r:<24}{acc:>10}{errs:>8}{fb:>10}{pct(ms, .5):>9.0f}{pct(ms, .95):>9.0f}")

    if len(args.routers) >= 2:
        a, b = args.routers[:2]
        agree = sum(1 for x, y in zip(results[a], results[b]) if same_model(x.get("candidate"), y.get("candidate")))
        print(f"\nagreement {a} vs {b}: {100 * agree / len(rows):.0f}%")
        if args.verbose:
            for i, (x, y) in enumerate(zip(results[a], results[b])):
                if not same_model(x.get("candidate"), y.get("candidate")):
                    exp = rows[i].get("expected", "?")
                    print(f"  [{i}] expected={exp} {a}={x.get('candidate') or x.get('error')} "
                          f"{b}={y.get('candidate') or y.get('error')} :: {rows[i]['prompt'][:70]!r}")

    if any("error" in x for res in results.values() for x in res):
        print("\nsample errors:")
        shown = 0
        for r, res in results.items():
            for x in res:
                if "error" in x and shown < 5:
                    print(f"  {r}: {x['error']}")
                    shown += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
