#!/usr/bin/env bash
# DG4c: collect fixture logits, fit T on train split, eval holdout ranking.
# Does not flip calibrated:true. Lab ports only.
# Exit 2 if holdout present and any ranking miss.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIX="${OPENJEV_FIXTURES:-${ROOT}/testdata/openjev/choice_fixtures_v0.json}"
MODEL="${DIFFUSION_GGUF:-/root/models/diffusiongemma/diffusiongemma-26B-A4B-it-Q4_K_M.gguf}"
NGL="${DIFFUSION_NGL:-8}"
PORT="${DIFFUSION_PORT:-18093}"
OUT="${OPENJEV_TEMP_OUT:-${ROOT}/testdata/openjev/temperature_v0.json}"

export LD_LIBRARY_PATH="/root/nvidia-host:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/go/bin:${PATH:-}"

if [[ -z "${ZEROLLAMA_DIFFUSION_SERVER_BIN:-}" ]]; then
  CAND="${ROOT}/vendor/llama-cpp-diffusion-dd0cf0445/build/bin/llama-diffusion-gemma-server"
  [[ -x "${CAND}" ]] || { echo "error: build diffusion server" >&2; exit 1; }
  ZEROLLAMA_DIFFUSION_SERVER_BIN="${CAND}"
fi

OWNED=0 PID=""
cleanup() { [[ "${OWNED}" == "1" && -n "${PID}" ]] && kill "${PID}" 2>/dev/null || true; }
trap cleanup EXIT

if ! curl -sf "http://127.0.0.1:${PORT}/props" >/dev/null 2>&1; then
  "${ZEROLLAMA_DIFFUSION_SERVER_BIN}" -m "${MODEL}" -ngl "${NGL}" -c 2048 \
    --host 127.0.0.1 --port "${PORT}" > /tmp/dg4b-fit.log 2>&1 &
  PID=$!; OWNED=1
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:${PORT}/props" >/dev/null 2>&1 && break
    kill -0 "${PID}" 2>/dev/null || { tail -30 /tmp/dg4b-fit.log >&2; exit 1; }
    sleep 2
  done
fi

python3 - "${FIX}" "http://127.0.0.1:${PORT}" "${OUT}" <<'PY'
import json, math, os, sys, urllib.request

fix, base, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
doc = json.load(open(fix))
opts = doc.get("options") or {}
seed = int(opts.get("seed", 42))
steps = int(opts.get("n_steps", 8))

def tokenize(s):
    req = urllib.request.Request(base + "/tokenize", data=json.dumps({"content": s}).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())["tokens"]

def pack_prompt(state, questions):
    # Minimal mirror of Go PackOpenJevPrompt (markers first — DG6).
    ids = sorted(questions)
    lines = [
        "You are a structured System-1 decision model. Given STATE and QUESTIONS, emit:",
        "1) One MARKER LINE per question (exact spelling; fill the value), then",
        "2) A JSON object mapping each question_id to an answer.",
        "",
        "Marker formats:",
        "choice: <<question_id>> <criteria_key>",
        'For type=choice: {"type":"choice","choice":"<one criteria key>"}',
        "Do not include confidence. No markdown, no preamble.",
        "",
        "STATE:",
        json.dumps(state),
        "",
        "QUESTIONS:",
    ]
    for qid in ids:
        lines.append(f"- id={qid} {json.dumps(questions[qid])}")
    lines.append("")
    lines.append("MARKER LINES (emit these first, filled):")
    for qid in ids:
        if questions[qid].get("type", "").lower() == "choice":
            lines.append(f"<<{qid}>> <criteria_key>")
    lines.append("")
    lines.append("Emit markers first, then JSON answers object:")
    return "\n".join(lines)

cases = []
for case in doc["cases"]:
    qs = case["questions"]
    # DG4/DG6 temperature fit is choice-only (noul/score use separate expect keys).
    gold = None
    for qid, exp in (case.get("expect") or {}).items():
        if "choice" in exp:
            gold = exp["choice"]
            break
    if gold is None:
        continue
    slots = []
    gather = []
    for qid in sorted(qs):
        q = qs[qid]
        if q.get("type", "").lower() != "choice":
            continue
        crit = q.get("criteria") or {}
        opts_ids = sorted(crit)
        marker = tokenize(f"<<{qid}>>")
        options = []
        for oid in opts_ids:
            toks = tokenize(oid)
            options.append({"option_id": oid, "tokens": toks})
            gather.append({"question_id": qid, "option_id": oid, "tokens": toks})
        slots.append({"question_id": qid, "marker_tokens": marker, "options": options})
    body = {
        "model": "openjev",
        "prompt": pack_prompt(case["state"], qs),
        "seed": seed,
        "n_steps": steps,
        "max_tokens": int(opts.get("max_tokens", 128)),
        "readout": {"slots": slots, "gather": gather},
    }
    req = urllib.request.Request(base + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read().decode())
    hits = resp.get("gather") or []
    # Prefer hits for the gold question when multi-q.
    qhits = [h for h in hits if h.get("question_id") in qs and qs[h["question_id"]].get("type", "").lower() == "choice"]
    if not qhits:
        qhits = hits
    qhits = sorted(qhits, key=lambda h: h["option_id"])
    option_ids = [h["option_id"] for h in qhits]
    logits = [float(h["logit_mean"]) for h in qhits]
    # argmax under T=1 for ranking check
    if logits:
        argmax = option_ids[max(range(len(logits)), key=lambda i: logits[i])]
    else:
        argmax = None
    cases.append({
        "id": case["id"],
        "split": case.get("split", "train"),
        "option_ids": option_ids,
        "logits": logits,
        "gold": gold,
        "argmax_t1": argmax,
        "hit_t1": argmax == gold,
        "readout_mode": resp.get("readout_mode"),
    })

def softmax_temp(logits, t):
    m = max(logits)
    xs = [math.exp((x - m) / t) for x in logits]
    s = sum(xs) or 1.0
    return [x / s for x in xs]

def fit_temp(subset):
    best_t, best_nll = 1.0, float("inf")
    for i in range(0, int((5.0 - 0.5) / 0.05) + 1):
        t = 0.5 + i * 0.05
        if t > 5.0 + 1e-9:
            break
        total = 0.0
        n = 0
        for c in subset:
            if c["gold"] not in c["option_ids"] or not c["logits"]:
                continue
            p = softmax_temp(c["logits"], t)
            gi = c["option_ids"].index(c["gold"])
            total += -math.log(max(p[gi], 1e-12))
            n += 1
        if n and total / n < best_nll:
            best_nll = total / n
            best_t = t
    return best_t, best_nll

train = [c for c in cases if c["split"] != "holdout"]
holdout = [c for c in cases if c["split"] == "holdout"]
if not train:
    train = cases
    holdout = []

best_t, best_nll = fit_temp(train)

holdout_rows = []
hold_ok = 0
for c in holdout:
    hit = bool(c.get("hit_t1"))
    hold_ok += int(hit)
    p = softmax_temp(c["logits"], best_t) if c["logits"] else []
    gold_p = p[c["option_ids"].index(c["gold"])] if c["gold"] in c["option_ids"] and p else None
    holdout_rows.append({
        "id": c["id"],
        "gold": c["gold"],
        "argmax": c["argmax_t1"],
        "hit": hit,
        "gold_prob_at_T": gold_p,
        "logits": c["logits"],
    })

hold_n = len(holdout)
hold_acc = (hold_ok / hold_n) if hold_n else None
# DG4c gate: holdout ranking must be perfect to even discuss calibrated:true — still do not flip it in code.
gate_ok = hold_n > 0 and hold_ok == hold_n

payload = {
    "version": "v0",
    "seed": seed,
    "n_steps": steps,
    "temperature": best_t,
    "train_nll": best_nll,
    "train_n": len(train),
    "holdout_n": hold_n,
    "holdout_accuracy": hold_acc,
    "holdout_gate_ok": gate_ok,
    "calibrated": False,
    "note": "DG4c: T fitted on train only; holdout_gate_ok means ranking matched labels. Still calibrated:false until operator sign-off + larger set.",
    "cases": cases,
}
open(out_path, "w").write(json.dumps(payload, indent=2) + "\n")

hold_path = out_path.replace("temperature_v0.json", "holdout_v0.json")
if hold_path == out_path:
    hold_path = out_path + ".holdout.json"
hold_doc = {
    "version": "v0",
    "temperature": best_t,
    "holdout_accuracy": hold_acc,
    "holdout_gate_ok": gate_ok,
    "calibrated": False,
    "cases": holdout_rows,
}
open(hold_path, "w").write(json.dumps(hold_doc, indent=2) + "\n")

# DG7: merge fixture_report (if present) into calib_gate_v0.json — never auto-flip calibrated.
report_path = fix.replace("choice_fixtures_v0.json", "fixture_report_v0.json")
fixture_n = fixture_acc = marker_rate = 0
if os.path.isfile(report_path):
    rep = json.load(open(report_path))
    fixture_n = int(rep.get("n") or 0)
    fixture_acc = float(rep.get("accuracy") or 0)
    marker_rate = float(rep.get("marker_rate") or 0)

MIN_HOLD_N, MIN_HOLD_ACC = 5, 0.9
MIN_FIX_ACC, MIN_MARK = 0.9, 0.9
failed = []
if hold_n < MIN_HOLD_N:
    failed.append("choice_holdout_n")
if (hold_acc or 0) < MIN_HOLD_ACC:
    failed.append("choice_holdout_accuracy")
if fixture_n == 0 or fixture_acc < MIN_FIX_ACC:
    failed.append("fixture_accuracy")
if marker_rate < MIN_MARK:
    failed.append("marker_rate")
gate_ready = len(failed) == 0
operator_signoff = False
calibrated = False  # never true from this script

gate_path = out_path.replace("temperature_v0.json", "calib_gate_v0.json")
if gate_path == out_path:
    gate_path = out_path + ".calib_gate.json"
gate_doc = {
    "version": "v0",
    "temperature": best_t,
    "choice_holdout_n": hold_n,
    "choice_holdout_accuracy": hold_acc,
    "holdout_gate_ok": gate_ok,
    "fixture_n": fixture_n,
    "fixture_accuracy": fixture_acc,
    "marker_rate": marker_rate,
    "operator_signoff": False,
    "gate_ready": gate_ready,
    "failed_checks": failed,
    "calibrated": calibrated,
    "note": "DG9: gate_ready = floors met. operator_signoff stays false (lab). Serve: CALIBRATED=1; noul also NOUL_CALIBRATED=1. Noul floors merged by fit_noul_bias.sh.",
}
# Preserve DG9 noul fields if already present (temperature fit is choice-only).
if os.path.isfile(gate_path):
    try:
        prev = json.load(open(gate_path))
        for k in ("noul_holdout_n", "noul_holdout_accuracy", "noul_bias"):
            if k in prev:
                gate_doc[k] = prev[k]
    except Exception:
        pass
open(gate_path, "w").write(json.dumps(gate_doc, indent=2) + "\n")

print(json.dumps({
    "temperature": best_t,
    "train_nll": best_nll,
    "train_n": len(train),
    "holdout_n": hold_n,
    "holdout_accuracy": hold_acc,
    "holdout_gate_ok": gate_ok,
    "gate_ready": gate_ready,
    "calibrated": False,
    "out": out_path,
    "holdout_out": hold_path,
    "calib_gate_out": gate_path,
}, indent=2))
# Soft signal: nonzero exit if holdout present and any miss (lab CI can opt in later).
if hold_n and hold_ok < hold_n:
    sys.exit(2)
PY
