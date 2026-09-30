#!/usr/bin/env bash
# DG4 accuracy harness: run labeled choice fixtures through Go OpenJev proxy.
# Reports hit rate; does NOT flip calibrated:true (DG4 gate incomplete).
# Lab ports only. Requires diffusion sibling + optional existing lab serve.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIX="${OPENJEV_FIXTURES:-${ROOT}/testdata/openjev/choice_fixtures_v0.json}"
MODEL="${DIFFUSION_GGUF:-/root/models/diffusiongemma/diffusiongemma-26B-A4B-it-Q4_K_M.gguf}"
NGL="${DIFFUSION_NGL:-8}"
DIFF_PORT="${DIFFUSION_PORT:-18093}"
GO_PORT="${ZEROLLAMA_LAB_PORT:-11435}"

export LD_LIBRARY_PATH="/root/nvidia-host:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/go/bin:${PATH:-}"

[[ -f "${FIX}" ]] || { echo "error: missing ${FIX}" >&2; exit 1; }
ZL="${ROOT}/zerollama"
[[ -x "${ZL}" ]] || { echo "error: missing ${ZL}" >&2; exit 1; }

if [[ -z "${ZEROLLAMA_DIFFUSION_SERVER_BIN:-}" ]]; then
  CAND="${ROOT}/vendor/llama-cpp-diffusion-dd0cf0445/build/bin/llama-diffusion-gemma-server"
  [[ -x "${CAND}" ]] || { echo "error: build diffusion server first" >&2; exit 1; }
  ZEROLLAMA_DIFFUSION_SERVER_BIN="${CAND}"
fi

DIFF_PID="" GO_PID="" OWNED_DIFF=0 OWNED_GO=0
cleanup() {
  [[ "${OWNED_GO}" == "1" && -n "${GO_PID}" ]] && kill "${GO_PID}" 2>/dev/null || true
  [[ "${OWNED_DIFF}" == "1" && -n "${DIFF_PID}" ]] && kill "${DIFF_PID}" 2>/dev/null || true
}
trap cleanup EXIT

if ! curl -sf "http://127.0.0.1:${DIFF_PORT}/props" >/dev/null 2>&1; then
  "${ZEROLLAMA_DIFFUSION_SERVER_BIN}" -m "${MODEL}" -ngl "${NGL}" -c 2048 \
    --host 127.0.0.1 --port "${DIFF_PORT}" > /tmp/dg4-fix-diff.log 2>&1 &
  DIFF_PID=$!; OWNED_DIFF=1
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:${DIFF_PORT}/props" >/dev/null 2>&1 && break
    kill -0 "${DIFF_PID}" 2>/dev/null || { tail -30 /tmp/dg4-fix-diff.log >&2; exit 1; }
    sleep 2
  done
fi

if ! curl -sf "http://127.0.0.1:${GO_PORT}/api/version" >/dev/null 2>&1; then
  OLLAMA_HOST="127.0.0.1:${GO_PORT}" \
  ZEROLLAMA_OPENJEV_URL="http://127.0.0.1:${DIFF_PORT}" \
  ZEROLLAMA_OPENJEV_CALIBRATED="${ZEROLLAMA_OPENJEV_CALIBRATED:-}" \
  ZEROLLAMA_OPENJEV_NOUL_CALIBRATED="${ZEROLLAMA_OPENJEV_NOUL_CALIBRATED:-}" \
  ZEROLLAMA_OPENJEV_NOUL_BIAS="${ZEROLLAMA_OPENJEV_NOUL_BIAS:-}" \
  ZEROLLAMA_OPENJEV_TEMP="${ZEROLLAMA_OPENJEV_TEMP:-}" \
  OLLAMA_NO_CLOUD=true OLLAMA_TRAINING=false ZEROLLAMA_RUNTIME_EMBED=0 \
    "${ZL}" serve > /tmp/dg4-acc-go.log 2>&1 &
  GO_PID=$!; OWNED_GO=1
  for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:${GO_PORT}/api/version" >/dev/null 2>&1 && break
    kill -0 "${GO_PID}" 2>/dev/null || { tail -30 /tmp/dg4-acc-go.log >&2; exit 1; }
    sleep 1
  done
elif [[ -n "${ZEROLLAMA_OPENJEV_CALIBRATED:-}${ZEROLLAMA_OPENJEV_NOUL_CALIBRATED:-}" ]]; then
  echo "warning: reusing Go :${GO_PORT}; OPENJEV_*CALIBRATED only applies if that process was started with them" >&2
fi

python3 - "${FIX}" "http://127.0.0.1:${GO_PORT}" <<'PY'
import json, sys, urllib.request

fix, base = sys.argv[1], sys.argv[2]
doc = json.load(open(fix))
opts = doc.get("options") or {}
model = doc.get("model", "openjev")
ok = fail = 0
marker_hit = 0
soft_rows = []
soft_n = soft_ok = 0
rows = []
for case in doc["cases"]:
    body = {
        "model": model,
        "state": case["state"],
        "questions": case["questions"],
        "options": opts,
    }
    req = urllib.request.Request(
        base + "/v1/systemone",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read().decode())
    expect = case.get("expect") or {}
    hit = True
    detail = {}
    any_marker = False
    gating = case.get("gate", True) is not False and case.get("split") != "report"
    for qid, exp in expect.items():
        ans = resp.get("answers", {}).get(qid)
        if isinstance(ans, str):
            ans = json.loads(ans)
        ans = ans or {}
        src = ans.get("score_source") or ""
        if src in ("answer_marker_logit", "hybrid_marker_gather"):
            any_marker = True
        if "choice" in exp:
            got = ans.get("choice")
            want = exp.get("choice")
            detail[qid] = {"got": got, "want": want, "score_source": src, "calibrated": ans.get("calibrated")}
            if got != want:
                hit = False
        elif "noul_min" in exp or "noul_max" in exp:
            got = ans.get("noul")
            detail[qid] = {"got_noul": got, "noul_min": exp.get("noul_min"), "noul_max": exp.get("noul_max"), "score_source": src, "calibrated": ans.get("calibrated")}
            if got is None:
                hit = False
            else:
                if "noul_min" in exp and float(got) < float(exp["noul_min"]):
                    hit = False
                if "noul_max" in exp and float(got) > float(exp["noul_max"]):
                    hit = False
        elif "score_min" in exp or "score_max" in exp:
            got = ans.get("score")
            detail[qid] = {"got_score": got, "score_min": exp.get("score_min"), "score_max": exp.get("score_max"), "score_source": src, "calibrated": ans.get("calibrated")}
            if got is None:
                hit = False
            else:
                if "score_min" in exp and float(got) < float(exp["score_min"]):
                    hit = False
                if "score_max" in exp and float(got) > float(exp["score_max"]):
                    hit = False
        else:
            detail[qid] = {"error": "unknown expect keys", "exp": exp}
            hit = False
        # DG9: noul calibrated:true only when ZEROLLAMA_OPENJEV_NOUL_CALIBRATED is on
        import os
        noul_cal_ok = os.environ.get("ZEROLLAMA_OPENJEV_NOUL_CALIBRATED", "") in ("1", "true", "on", "TRUE", "ON")
        if ans.get("calibrated") is True and (
            ans.get("type") == "noul" or "noul_min" in exp or "noul_max" in exp
        ) and not noul_cal_ok:
            raise SystemExit(f"BUG: calibrated:true on OpenJev noul without NOUL_CALIBRATED ({case['id']})")
    rows.append({"id": case["id"], "ok": hit, "marker": any_marker, "gating": gating, **detail})
    if gating:
        ok += int(hit)
        fail += int(not hit)
        marker_hit += int(any_marker)
    else:
        soft_rows.append({"id": case["id"], "ok": hit, **detail})
        soft_n += 1
        soft_ok += int(hit)

n = ok + fail
report = {
    "ok": ok,
    "fail": fail,
    "n": n,
    "accuracy": (ok / n if n else 0),
    "marker_cases": marker_hit,
    "marker_rate": (marker_hit / n if n else 0),
    "calibrated": False,
    "report_noul": {"n": soft_n, "ok": soft_ok, "cases": soft_rows},
    "cases": rows,
}
print(json.dumps(report, indent=2))
report_path = fix.replace("choice_fixtures_v0.json", "fixture_report_v0.json")
if report_path == fix:
    report_path = fix + ".report.json"
open(report_path, "w").write(json.dumps(report, indent=2) + "\n")
# DG4a: harness always exits 0 after reporting — accuracy gate is operator-owned until DG4 calib.
# Soft floor: warn on stderr if accuracy is 0.
if n and ok == 0:
    print("warning: 0/{n} fixtures matched expected labels".format(n=n), file=sys.stderr)
if n and marker_hit == 0:
    print("warning: 0/{n} cases used marker/hybrid score_source (all gather fallback)".format(n=n), file=sys.stderr)
if soft_n and soft_ok < soft_n:
    print("note: report-only (non-gating) cases {ok}/{n}".format(ok=soft_ok, n=soft_n), file=sys.stderr)
PY
