#!/usr/bin/env python3
"""AR7 router evaluation harness (docs/auto-router.md §13, findings: docs/auto-router-findings.md).

Replays testdata/router_eval/prompts.jsonl through POST /v1/route (dry runs, route:"fresh"
so sticky / cache memory never skews results) and scores each router:

  capability   routed model has every capability the request needs (must be 100%)
  tier         chosen candidate tier vs the row's expected minimum tier (router.yaml tiers)
  strength     row task ∈ chosen candidate strengths (router.yaml strengths)
  latency      server decision latency p50 / p95
  quality      optional: --judge MODEL --baseline MODEL generates real answers from the
               routed model and a fixed baseline, and a judge model picks A / B / TIE

Gate (AR5 leaves opt-in only on PASS): capability 100% of servable rows, decision p50 ≤
--gate-ms (default 50), and with a quality run: routed wins+ties ≥ 50% and wins ≥ losses.
Without --judge the gate is INCOMPLETE — routing accuracy alone cannot show quality.

    # dry-run scoring only (safe; nothing generates)
    ./scripts/eval/router_eval.py --router auto --router smart testdata/router_eval/prompts.jsonl

    # full gate on a lab node
    ./scripts/eval/router_eval.py --router smart --judge qwen3:32b --baseline qwen3:8b \\
        --quality-sample 40 --json /tmp/router_eval.json testdata/router_eval/prompts.jsonl

    ./scripts/eval/router_eval.py --check testdata/router_eval/prompts.jsonl   # schema only

Stdlib only. Default host is the lab port — never point agent work at production :11434.
"""

import argparse
import base64
import json
import random
import sys
import time
import urllib.error
import urllib.request

TINY_PNG = base64.b64encode(bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082")).decode()
FAKE_MEDIA = base64.b64encode(b"router-eval-placeholder").decode()  # /v1/route never decodes media

TOOL = {"type": "function", "function": {
    "name": "lookup", "description": "Look something up",
    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}

REQUIRED_CAPS = {"image": {"vision"}, "video": {"vision", "video"}, "audio": {"audio"}, "tools": {"tools"}}
TIER_ORDER = {"low": 0, "mid": 1, "high": 2}
ENUMS = {
    "surface": {"chat", "generate", "image", "image_edit"},
    "task": {"code", "math", "chat", "extract", "creative"},
    "tier": set(TIER_ORDER),
    "output": {"text", "image", "speech", "video", "music"},
}


def http_json(host, method, path, body=None, timeout=60.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(host.rstrip("/") + path, data=data, method=method,
                                 headers={"content-type": "application/json"})
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.load(resp), (time.monotonic() - start) * 1000
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read())
        except ValueError:
            payload = {"error": f"HTTP {e.code}"}
        return e.code, payload, (time.monotonic() - start) * 1000


def norm(name):
    """'qwen3' → 'qwen3:latest'; drop the default library prefix."""
    if not name:
        return ""
    name = name.removeprefix("registry.ollama.ai/").removeprefix("library/")
    return name if ":" in name else name + ":latest"


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            for k, allowed in ENUMS.items():
                if row.get(k) not in allowed:
                    raise SystemExit(f"line {n}: {k}={row.get(k)!r} not in {sorted(allowed)}")
            rows.append(row)
    return rows


def messages_for(row):
    msgs = []
    if row.get("system"):
        msgs.append({"role": "system", "content": row["system"]})
    user = {"role": "user", "content": row["prompt"]}
    att = set(row.get("attach", []))
    if "image" in att:
        user["images"] = [TINY_PNG]
    if "video" in att:
        user["videos"] = [FAKE_MEDIA]
    if "audio" in att:
        user["audio_clips"] = [FAKE_MEDIA]
    msgs.append(user)
    return msgs


def route_body(router, row):
    body = {"model": router, "options": {"zerollama": {"route": "fresh"}}}
    s = row["surface"]
    if s == "chat":
        body["messages"] = messages_for(row)
        if "tools" in row.get("attach", []):
            body["tools"] = [TOOL]
    elif s == "generate":
        body["prompt"] = row["prompt"]
    else:
        body.update({"surface": s, "prompt": row["prompt"], "width": 512, "height": 512})
        if s == "image_edit":
            body["images"] = [TINY_PNG]
    return body


def required_caps(row):
    if row["surface"] in ("image", "image_edit"):
        return {"image"}
    caps = {"completion"}
    for a in row.get("attach", []):
        caps |= REQUIRED_CAPS[a]
    return caps


def pct(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def judge_pair(host, judge, question, routed, baseline, rng, timeout):
    """Returns 'win' | 'loss' | 'tie' for the routed answer (order randomised)."""
    flip = rng.random() < 0.5
    a, b = (baseline, routed) if flip else (routed, baseline)
    prompt = (f"Question:\n{question}\n\nAnswer A:\n{a}\n\nAnswer B:\n{b}\n\n"
              "Which answer is better (correct, complete, clear)? Reply with exactly one word: A, B or TIE.")
    status, out, _ = http_json(host, "POST", "/api/chat", {
        "model": judge, "stream": False, "options": {"temperature": 0, "num_predict": 8},
        "messages": [{"role": "system", "content": "You are an impartial judge."},
                     {"role": "user", "content": prompt}]}, timeout)
    verdict = (out.get("message", {}).get("content", "") if status == 200 else "").strip().upper()
    if verdict.startswith("TIE") or verdict[:1] not in ("A", "B"):
        return "tie"
    routed_won = (verdict[0] == "B") if flip else (verdict[0] == "A")
    return "win" if routed_won else "loss"


def answer(host, model, row, timeout):
    status, out, ms = http_json(host, "POST", "/api/chat", {
        "model": model, "stream": False, "messages": messages_for(row),
        "options": {"num_predict": 512, "zerollama": {"route": "fresh"}}}, timeout)
    return (out.get("message", {}).get("content", "") if status == 200 else None), ms


def evaluate(args, rows, router, inv):
    caps_by_model = {norm(c["name"]): set(c.get("capabilities") or []) for c in inv.get("candidates", [])}
    summary = inv.get("routers", {}).get(router)
    cand_policy = {norm(c["model"]): c for c in (summary or {}).get("candidates", [])}
    has_tiers = any(c.get("tier") for c in cand_policy.values())
    has_strengths = any(c.get("strengths") for c in cand_policy.values())

    m = {"router": router, "n": 0, "errors": 0, "unservable": 0, "cap_ok": 0, "cap_rows": 0,
         "cap_failures": [], "tier": {"exact": 0, "under": 0, "over": 0, "scored": 0},
         "strength": {"hit": 0, "scored": 0}, "escalated": 0, "latency_ms": [], "rows": [],
         "suggest": {"cross": 0, "hit": 0, "text": 0, "false_pos": 0}}
    for row in rows:
        status, out, wall = http_json(args.host, "POST", "/v1/route", route_body(router, row), args.timeout)
        m["n"] += 1
        need = required_caps(row)
        servable = any(need <= caps for caps in caps_by_model.values())
        sug = (out.get("suggestion") or {}).get("modality")
        rec = {"id": row["id"], "status": status, "candidate": out.get("candidate"), "suggest": sug,
               "reason": out.get("reason"), "escalated": bool(out.get("escalated")), "wall_ms": round(wall, 1)}
        m["rows"].append(rec)
        if status != 200:
            if not servable:
                m["unservable"] += 1  # no local model can do this — not a routing error
            else:
                m["errors"] += 1
                rec["error"] = out.get("error")
            continue
        m["latency_ms"].append(out.get("latency_ms", wall))
        chosen = norm(out.get("candidate"))
        if need - {"completion"}:
            m["cap_rows"] += 1
            if need <= caps_by_model.get(chosen, set()):
                m["cap_ok"] += 1
            else:
                m["cap_failures"].append(f"{row['id']}: {chosen} lacks {sorted(need - caps_by_model.get(chosen, set()))}")
        m["escalated"] += rec["escalated"]
        if row["surface"] == "chat":  # AR8 cross-modal advisory: recall + false positives
            sg = m["suggest"]
            if row["output"] == "text":
                sg["text"] += 1
                sg["false_pos"] += bool(sug)
            else:
                sg["cross"] += 1
                sg["hit"] += sug == row["output"]
        pol = cand_policy.get(chosen)
        text_row = row["output"] == "text" and row["surface"] in ("chat", "generate")
        if has_tiers and text_row and pol and pol.get("tier") in TIER_ORDER:
            m["tier"]["scored"] += 1
            d = TIER_ORDER[pol["tier"]] - TIER_ORDER[row["tier"]]
            m["tier"]["exact" if d == 0 else ("over" if d > 0 else "under")] += 1
        if has_strengths and text_row and pol is not None:
            m["strength"]["scored"] += 1
            labels = {l for ls in (pol.get("strengths") or {}).values() for l in ls}
            m["strength"]["hit"] += row["task"] in labels

    if args.judge and args.baseline:
        rng = random.Random(args.seed)
        pool = [(row, rec) for row, rec in zip(rows, m["rows"])
                if rec["status"] == 200 and row["surface"] == "chat" and row["output"] == "text"
                and not row.get("attach")]
        rng.shuffle(pool)
        q = {"win": 0, "tie": 0, "loss": 0, "same_model": 0, "failed": 0}
        for row, rec in pool[:args.quality_sample]:
            if norm(rec["candidate"]) == norm(args.baseline):
                q["tie"] += 1
                q["same_model"] += 1
                continue
            routed, _ = answer(args.host, rec["candidate"], row, args.timeout)
            base, _ = answer(args.host, args.baseline, row, args.timeout)
            if routed is None or base is None:
                q["failed"] += 1
                continue
            q[judge_pair(args.host, args.judge, row["prompt"], routed, base, rng, args.timeout)] += 1
        m["quality"] = q
    return m


def gate(m, gate_ms):
    reasons, fail = [], False
    servable_caps = m["cap_rows"]
    if servable_caps and m["cap_ok"] < servable_caps:
        fail = True
        reasons.append(f"capability {m['cap_ok']}/{servable_caps}")
    p50 = pct(m["latency_ms"], 0.5)
    if p50 > gate_ms:
        fail = True
        reasons.append(f"p50 {p50:.0f}ms > {gate_ms}ms")
    if m["errors"]:
        reasons.append(f"{m['errors']} routing errors")
    q = m.get("quality")
    if not q:
        return ("FAIL" if fail else "INCOMPLETE"), reasons + ["no quality run (--judge/--baseline)"]
    judged = q["win"] + q["tie"] + q["loss"]
    if judged == 0:
        return "FAIL" if fail else "INCOMPLETE", reasons + ["quality run judged nothing"]
    if (q["win"] + q["tie"]) / judged < 0.5 or q["win"] < q["loss"]:
        fail = True
        reasons.append(f"quality W/T/L {q['win']}/{q['tie']}/{q['loss']}")
    return ("FAIL" if fail else "PASS"), reasons


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--router", action="append", default=[], help="router name (repeatable); default auto")
    ap.add_argument("--host", default="http://127.0.0.1:11435")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--judge", help="judge model for the quality run")
    ap.add_argument("--baseline", help="fixed baseline model the routed answers are compared to")
    ap.add_argument("--quality-sample", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gate-ms", type=float, default=50.0)
    ap.add_argument("--json", help="write per-row results and metrics here")
    ap.add_argument("--check", action="store_true", help="validate the dataset and exit")
    args = ap.parse_args()

    rows = load_rows(args.dataset)
    if args.check:
        print(f"ok: {len(rows)} rows")
        return 0
    if bool(args.judge) != bool(args.baseline):
        ap.error("--judge and --baseline go together")
    routers = args.router or ["auto"]

    status, inv, _ = http_json(args.host, "GET", "/api/router/inventory", timeout=args.timeout)
    if status != 200:
        sys.exit(f"inventory: HTTP {status} {inv} — is {args.host} a zerollama with AR2+?")

    results, worst = [], 0
    print(f"{len(rows)} prompts · {len(inv.get('candidates', []))} local models · {args.host}\n")
    print(f"{'router':<16}{'err':>5}{'unsrv':>7}{'caps':>10}{'tier=':>7}{'under':>7}{'over':>6}"
          f"{'strength':>10}{'esc':>5}{'p50ms':>7}{'p95ms':>7}  gate")
    for r in routers:
        m = evaluate(args, rows, r, inv)
        verdict, reasons = gate(m, args.gate_ms)
        m["gate"] = {"verdict": verdict, "reasons": reasons}
        results.append(m)
        worst = max(worst, {"PASS": 0, "INCOMPLETE": 0, "FAIL": 1}[verdict])
        t, st = m["tier"], m["strength"]
        tier_s = lambda k: f"{100 * t[k] / t['scored']:.0f}%" if t["scored"] else "-"
        caps = f"{m['cap_ok']}/{m['cap_rows']}" if m["cap_rows"] else "-"
        strength = f"{100 * st['hit'] / st['scored']:.0f}%" if st["scored"] else "-"
        print(f"{r:<16}{m['errors']:>5}{m['unservable']:>7}{caps:>10}{tier_s('exact'):>7}{tier_s('under'):>7}"
              f"{tier_s('over'):>6}{strength:>10}{m['escalated']:>5}{pct(m['latency_ms'], .5):>7.0f}"
              f"{pct(m['latency_ms'], .95):>7.0f}  {verdict}")
        if "quality" in m:
            q = m["quality"]
            print(f"{'':<16}quality vs {args.baseline} (judge {args.judge}): "
                  f"W/T/L {q['win']}/{q['tie']}/{q['loss']} (same model {q['same_model']}, failed {q['failed']})")
        sg = m["suggest"]
        if sg["cross"] or sg["text"]:
            print(f"{'':<16}cross-modal suggest: recall {sg['hit']}/{sg['cross']}, "
                  f"false positives {sg['false_pos']}/{sg['text']}")
        for why in reasons:
            print(f"{'':<16}- {why}")
        for f in m["cap_failures"][:5]:
            print(f"{'':<16}! {f}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"host": args.host, "dataset": args.dataset, "results": results}, f, indent=2)
        print(f"\nwrote {args.json}")
    return worst


if __name__ == "__main__":
    sys.exit(main())
