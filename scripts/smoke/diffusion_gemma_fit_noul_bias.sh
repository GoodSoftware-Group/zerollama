#!/usr/bin/env bash
# DG9: fit OpenJev noul positive-class logit bias on train fixtures; eval holdout.
# Writes testdata/openjev/noul_bias_v0.json. Does not flip calibrated by itself.
# Lab ports only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIX="${OPENJEV_FIXTURES:-${ROOT}/testdata/openjev/choice_fixtures_v0.json}"
MODEL="${DIFFUSION_GGUF:-/root/models/diffusiongemma/diffusiongemma-26B-A4B-it-Q4_K_M.gguf}"
NGL="${DIFFUSION_NGL:-8}"
PORT="${DIFFUSION_PORT:-18093}"
OUT="${OPENJEV_NOUL_BIAS_OUT:-${ROOT}/testdata/openjev/noul_bias_v0.json}"
TEMP="${ZEROLLAMA_OPENJEV_TEMP:-0.5}"

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
    --host 127.0.0.1 --port "${PORT}" > /tmp/dg9-noul-fit.log 2>&1 &
  PID=$!; OWNED=1
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:${PORT}/props" >/dev/null 2>&1 && break
    kill -0 "${PID}" 2>/dev/null || { tail -30 /tmp/dg9-noul-fit.log >&2; exit 1; }
    sleep 2
  done
fi

python3 - "${FIX}" "http://127.0.0.1:${PORT}" "${OUT}" "${TEMP}" <<'PY'
import json, math, os, sys, urllib.request

fix, base, out_path, temp_s = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
temp = float(temp_s)
doc = json.load(open(fix))
opts = doc.get("options") or {}
seed = int(opts.get("seed", 42))
steps = int(opts.get("n_steps", 8))
max_tok = int(opts.get("max_tokens", 128))

def tokenize(s):
    req = urllib.request.Request(base + "/tokenize", data=json.dumps({"content": s}).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())["tokens"]

def pack_prompt(state, questions):
    ids = sorted(questions)
    lines = [
        "You are a structured System-1 decision model. Given STATE and QUESTIONS, emit:",
        "1) One MARKER LINE per question (exact spelling; fill the value), then",
        "2) A JSON object mapping each question_id to an answer.",
        "",
        "Marker formats:",
        "noul: <<question_id>> 0|1",
        'For type=noul: {"type":"noul","noul":0.0-1.0}  (marker 1 = statement holds / immediate action required; 0 otherwise)',
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
        if questions[qid].get("type", "").lower() == "noul":
            lines.append(f"<<{qid}>> 0|1")
    lines.append("")
    lines.append("Emit markers first, then JSON answers object:")
    return "\n".join(lines)

def softmax_temp(logits, t):
    m = max(logits)
    xs = [math.exp((x - m) / t) for x in logits]
    s = sum(xs) or 1.0
    return [x / s for x in xs]

cases = []
for case in doc["cases"]:
    qs = case["questions"]
    exp = case.get("expect") or {}
    gold_true = None
    for qid, e in exp.items():
        if qs.get(qid, {}).get("type", "").lower() != "noul":
            continue
        if "noul_min" in e:
            gold_true = True
            break
        if "noul_max" in e:
            gold_true = False
            break
    if gold_true is None:
        continue
    slots = []
    gather = []
    for qid in sorted(qs):
        if qs[qid].get("type", "").lower() != "noul":
            continue
        marker = tokenize(f"<<{qid}>>")
        options = []
        for oid in ("0", "1"):
            toks = tokenize(oid)
            options.append({"option_id": oid, "tokens": toks})
            gather.append({"question_id": qid, "option_id": oid, "tokens": toks})
        slots.append({"question_id": qid, "marker_tokens": marker, "options": options})
    body = {
        "model": "openjev",
        "prompt": pack_prompt(case["state"], qs),
        "seed": seed,
        "n_steps": steps,
        "max_tokens": max_tok,
        "readout": {"slots": slots, "gather": gather},
    }
    req = urllib.request.Request(base + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read().decode())
    hits = {h["option_id"]: float(h["logit_mean"]) for h in (resp.get("gather") or []) if h.get("question_id")}
    # prefer hits for the noul qid
    by = {}
    for h in resp.get("gather") or []:
        by.setdefault(h["question_id"], {})[h["option_id"]] = float(h["logit_mean"])
    lf = lt = None
    for qid, opts_m in by.items():
        if "0" in opts_m and "1" in opts_m:
            lf, lt = opts_m["0"], opts_m["1"]
            break
    if lf is None:
        lf, lt = hits.get("0"), hits.get("1")
    cases.append({
        "id": case["id"],
        "split": case.get("split", "train"),
        "logit_false": lf,
        "logit_true": lt,
        "gold_true": gold_true,
        "readout_mode": resp.get("readout_mode"),
        "margin": (None if lf is None or lt is None else lt - lf),
    })

train = [c for c in cases if c["split"] != "holdout" and c["logit_false"] is not None]
holdout = [c for c in cases if c["split"] == "holdout" and c["logit_false"] is not None]

def fit_bias(subset):
    best_b, best_nll = 0.0, float("inf")
    for i in range(0, int(20 / 0.25) + 1):
        b = i * 0.25
        total = 0.0
        n = 0
        for c in subset:
            p = softmax_temp([c["logit_false"], c["logit_true"] - b], temp)
            prob = p[1] if c["gold_true"] else p[0]
            total += -math.log(max(prob, 1e-12))
            n += 1
        if n and total / n < best_nll:
            best_nll = total / n
            best_b = b
    return best_b, best_nll

best_b, best_nll = fit_bias(train) if train else (0.0, None)

def eval_split(subset, b):
    ok = 0
    rows = []
    for c in subset:
        p = softmax_temp([c["logit_false"], c["logit_true"] - b], temp)
        noul = p[1]
        pred = noul >= 0.5
        hit = pred == c["gold_true"]
        ok += int(hit)
        rows.append({**c, "noul_at_bias": noul, "pred_true": pred, "hit": hit})
    n = len(subset)
    return (ok / n if n else None), ok, n, rows

hold_acc, hold_ok, hold_n, hold_rows = eval_split(holdout, best_b)
train_acc, train_ok, train_n, _ = eval_split(train, best_b)
gate_ok = hold_n > 0 and hold_ok == hold_n

payload = {
    "version": "v0",
    "temperature": temp,
    "noul_bias": best_b,
    "train_nll": best_nll,
    "train_n": train_n,
    "train_accuracy": train_acc,
    "holdout_n": hold_n,
    "holdout_accuracy": hold_acc,
    "holdout_gate_ok": gate_ok,
    "calibrated": False,
    "note": "DG9: bias fit on train noul only. Set ZEROLLAMA_OPENJEV_NOUL_BIAS. Noul calibrated needs OPENJEV_CALIBRATED + OPENJEV_NOUL_CALIBRATED.",
    "cases": cases,
    "holdout": hold_rows,
}
open(out_path, "w").write(json.dumps(payload, indent=2) + "\n")

# Merge noul floors into calib_gate_v0.json (preserve choice floors from fit_temperature).
gate_path = out_path.replace("noul_bias_v0.json", "calib_gate_v0.json")
if gate_path == out_path:
    gate_path = out_path + ".calib_gate.json"
gate = {}
if os.path.isfile(gate_path):
    try:
        gate = json.load(open(gate_path))
    except Exception:
        gate = {}
gate.setdefault("version", "v0")
gate["noul_bias"] = best_b
gate["noul_holdout_n"] = hold_n
gate["noul_holdout_accuracy"] = hold_acc
gate["note"] = (
    "DG9: gate_ready = choice+noul+marker floors. operator_signoff stays false (lab). "
    "Serve: CALIBRATED=1; noul also NOUL_CALIBRATED=1."
)
# Recompute failed_checks / gate_ready with noul floors when choice floors already present.
failed = list(gate.get("failed_checks") or [])
failed = [f for f in failed if not f.startswith("noul_")]
MIN_NOUL_N, MIN_ACC = 2, 0.9
if hold_n < MIN_NOUL_N:
    failed.append("noul_holdout_n")
elif (hold_acc or 0) < MIN_ACC:
    failed.append("noul_holdout_accuracy")
# Drop stale choice fails only when we cannot re-eval; keep existing choice fields.
gate["failed_checks"] = failed
# gate_ready: true only if no fails left (choice fields must already be ok from temp fit)
choice_ok = (
    int(gate.get("choice_holdout_n") or 0) >= 5
    and float(gate.get("choice_holdout_accuracy") or 0) >= 0.9
    and int(gate.get("fixture_n") or 0) > 0
    and float(gate.get("fixture_accuracy") or 0) >= 0.9
    and float(gate.get("marker_rate") or 0) >= 0.9
)
gate["gate_ready"] = choice_ok and len(failed) == 0
gate["calibrated"] = False
gate.setdefault("operator_signoff", False)
open(gate_path, "w").write(json.dumps(gate, indent=2) + "\n")

print(json.dumps({
    "noul_bias": best_b,
    "train_nll": best_nll,
    "train_accuracy": train_acc,
    "holdout_accuracy": hold_acc,
    "holdout_gate_ok": gate_ok,
    "out": out_path,
    "calib_gate_out": gate_path,
    "gate_ready": gate.get("gate_ready"),
}, indent=2))
if hold_n and hold_ok < hold_n:
    sys.exit(2)
PY
