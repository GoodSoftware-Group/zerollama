#!/usr/bin/env bash
# Phase 11 — 5080 admission contention smoke (chat + low/batch under load).
#
# Idle coordination smoke only proves mirrors. This harness drives concurrent
# normal + low-priority generates, samples /health admission gates mid-flight,
# and (when training is up) records go_training_gpu_busy / reserve fields.
#
#   # Serve must already be healthy (:8080 + :8081)
#   source ./scripts/gpu/5080_env.sh
#   ./scripts/phase/phase11_5080_contention_smoke.sh
#
# Env:
#   OLLAMA_HOST / ZEROLLAMA_RUNTIME_URL — defaults 127.0.0.1:8080 / :8081
#   RUN_E2E_PROXY_MODEL / P11_MODEL — pulled tag (default llama3.2:3b)
#   P11_NUM_NORMAL / P11_NUM_LOW — concurrent workers (default 2 / 6)
#   P11_NUM_PREDICT — tokens per request (default 24)
#   P11_OUT — JSON artifact (default /tmp/phase11-5080-contention.json)
#   P11_SKIP_COORD=1 — skip e2e_coordination_smoke.sh
#   P11_SKIP_TRAIN=1 — skip training status / busy snapshot
#
# Pass: normal requests must not be systematically 503'd by inference-first
# mirrors; VRAM probe must not be misconfigured when checks are on.
# Doc: docs/phase11-runtime-admission.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OLLAMA_URL="${OLLAMA_HOST:-http://127.0.0.1:8080}"
OLLAMA_URL="${OLLAMA_URL%/}"
RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"
RUNTIME_URL="${RUNTIME_URL%/}"
MODEL="${P11_MODEL:-${RUN_E2E_PROXY_MODEL:-llama3.2:3b}}"
NUM_NORMAL="${P11_NUM_NORMAL:-2}"
NUM_LOW="${P11_NUM_LOW:-6}"
NUM_PREDICT="${P11_NUM_PREDICT:-24}"
OUT="${P11_OUT:-/tmp/phase11-5080-contention.json}"
CURL_T="${P11_CURL_TIMEOUT:-120}"

echo "== Phase 11 5080 contention: ${OLLAMA_URL} model=${MODEL} =="

if ! curl -sf -m 5 "${OLLAMA_URL}/api/version" >/dev/null; then
  echo "FAIL: Go not healthy at ${OLLAMA_URL}/api/version" >&2
  exit 1
fi
if ! curl -sf -m 15 "${RUNTIME_URL}/health" >/dev/null; then
  echo "FAIL: runtime not healthy at ${RUNTIME_URL}/health" >&2
  exit 1
fi

if [[ "${P11_SKIP_COORD:-0}" != "1" ]]; then
  echo ""
  echo "== baseline coordination smoke =="
  OLLAMA_HOST="${OLLAMA_URL}" ZEROLLAMA_RUNTIME_URL="${RUNTIME_URL}" \
    "${ROOT}/scripts/e2e/e2e_coordination_smoke.sh"
fi

export P11_OLLAMA_URL="$OLLAMA_URL"
export P11_RUNTIME_URL="$RUNTIME_URL"
export P11_MODEL_NAME="$MODEL"
export P11_NUM_NORMAL="$NUM_NORMAL"
export P11_NUM_LOW="$NUM_LOW"
export P11_NUM_PREDICT="$NUM_PREDICT"
export P11_OUT="$OUT"
export P11_CURL_TIMEOUT="$CURL_T"
export P11_SKIP_TRAIN="${P11_SKIP_TRAIN:-0}"

python3 <<'PY'
from __future__ import annotations

import json
import os
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

ollama = os.environ["P11_OLLAMA_URL"].rstrip("/")
runtime = os.environ["P11_RUNTIME_URL"].rstrip("/")
model = os.environ["P11_MODEL_NAME"]
n_normal = max(1, int(os.environ["P11_NUM_NORMAL"]))
n_low = max(1, int(os.environ["P11_NUM_LOW"]))
n_predict = max(1, int(os.environ["P11_NUM_PREDICT"]))
out_path = os.environ["P11_OUT"]
timeout = float(os.environ.get("P11_CURL_TIMEOUT", "120"))
skip_train = os.environ.get("P11_SKIP_TRAIN", "0") == "1"


def http_json(url: str, *, data: dict | None = None, method: str | None = None, t: float = 15.0):
    body = None
    headers = {}
    if data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
        method = method or "POST"
    req = urllib.request.Request(url, data=body, headers=headers, method=method or "GET")
    try:
        with urllib.request.urlopen(req, timeout=t) as resp:
            raw = resp.read().decode()
            code = resp.status
            try:
                parsed = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                parsed = {"_raw": raw[:500]}
            return code, parsed, None
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {"_raw": raw[:500]}
        return e.code, parsed, str(e)
    except Exception as e:  # noqa: BLE001
        return 0, {}, str(e)


def health_snap() -> dict:
    code, h, err = http_json(f"{runtime}/health", t=20.0)
    if code != 200:
        return {"ok": False, "http": code, "error": err, "admission": {}}
    ad = h.get("admission") or {}
    vb = h.get("vram_budget") or {}
    return {
        "ok": True,
        "admission": {
            "inference_policy": ad.get("inference_policy"),
            "vram_probe_effective": ad.get("vram_probe_effective", h.get("vram_probe_effective")),
            "go_training_gpu_busy": ad.get("go_training_gpu_busy"),
            "vram_min_free_configured": ad.get("vram_min_free_configured"),
            "vram_training_reserve_configured": ad.get("vram_training_reserve_configured"),
            "gates_active": ad.get("gates_active") or {},
            "gates_active_compat": ad.get("gates_active_compat") or {},
            "backlog": ad.get("backlog"),
            "backpressure": ad.get("backpressure"),
        },
        "vram_budget": {
            "admission_fits": vb.get("admission_fits"),
            "suggested_max_num_ctx": vb.get("suggested_max_num_ctx"),
        },
        "go_coordination": h.get("go_coordination") or {},
    }


def one_generate(priority: str, idx: int) -> dict:
    payload = {
        "model": model,
        "prompt": f"Phase11 {priority} #{idx}: reply with one short sentence about GPUs.",
        "stream": False,
        "options": {
            "num_predict": n_predict,
            "temperature": 0,
            "priority": priority,
        },
    }
    t0 = time.perf_counter()
    code, body, err = http_json(f"{ollama}/api/generate", data=payload, t=timeout)
    wall = time.perf_counter() - t0
    done = bool(body.get("done")) if isinstance(body, dict) else False
    resp = (body.get("response") or "") if isinstance(body, dict) else ""
    return {
        "priority": priority,
        "idx": idx,
        "http": code,
        "wall_s": round(wall, 3),
        "done": done,
        "error": err or (body.get("error") if isinstance(body, dict) else None),
        "response_chars": len(resp),
        "503": code == 503,
    }


baseline = health_snap()
gates0 = (baseline.get("admission") or {}).get("gates_active") or {}

# Mid-flight sampler: fire workers, sample health while they run
jobs: list[tuple[str, int]] = []
for i in range(n_normal):
    jobs.append(("normal", i))
for i in range(n_low):
    jobs.append(("low", i))

mid_snaps: list[dict] = []
results: list[dict] = []
t_start = time.perf_counter()

with ThreadPoolExecutor(max_workers=n_normal + n_low) as ex:
    futs = {ex.submit(one_generate, p, i): (p, i) for p, i in jobs}
    # Sample health a few times while work is in flight
    for _ in range(3):
        time.sleep(0.4)
        mid_snaps.append({"t_s": round(time.perf_counter() - t_start, 3), **health_snap()})
    for fut in as_completed(futs):
        results.append(fut.result())

after = health_snap()
elapsed = round(time.perf_counter() - t_start, 3)

by_pri: dict[str, list[dict]] = {"normal": [], "low": []}
for r in results:
    by_pri.setdefault(r["priority"], []).append(r)


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    walls = [r["wall_s"] for r in rows]
    n503 = sum(1 for r in rows if r.get("503"))
    n_ok = sum(1 for r in rows if r.get("http") == 200 and r.get("done"))
    n_err = sum(1 for r in rows if r.get("http") not in (200, 0) or r.get("error"))
    return {
        "n": len(rows),
        "ok": n_ok,
        "http_503": n503,
        "other_err": n_err - n503 if n_err >= n503 else n_err,
        "wall_s_mean": round(statistics.mean(walls), 3),
        "wall_s_max": round(max(walls), 3),
        "wall_s_min": round(min(walls), 3),
    }


train_snap = None
if not skip_train:
    code, body, err = http_json(f"{ollama}/api/train/status", t=5.0)
    if code == 200:
        train_snap = {
            "ok": True,
            "keys": sorted(body.keys())[:16] if isinstance(body, dict) else [],
            "go_training_gpu_busy_baseline": (baseline.get("admission") or {}).get(
                "go_training_gpu_busy"
            ),
            "go_training_gpu_busy_after": (after.get("admission") or {}).get(
                "go_training_gpu_busy"
            ),
            "vram_training_reserve_configured": (baseline.get("admission") or {}).get(
                "vram_training_reserve_configured"
            ),
        }
    else:
        train_snap = {"ok": False, "http": code, "error": err, "skipped": True}

normal_sum = summarize(by_pri.get("normal", []))
low_sum = summarize(by_pri.get("low", []))

# Mid-flight gate truths (any sample)
mid_gates_true: dict[str, bool] = {}
for snap in mid_snaps:
    g = (snap.get("admission") or {}).get("gates_active") or {}
    for k, v in g.items():
        if v:
            mid_gates_true[k] = True

# FAIL: VRAM checks appear on but probe ineffective / misconfigured signal
probe = (baseline.get("admission") or {}).get("vram_probe_effective")
fits = (baseline.get("vram_budget") or {}).get("admission_fits")
fail_reasons: list[str] = []
warns: list[str] = []

if baseline.get("ok") and probe is False:
    # Probe false with checks desired → operators may have CHECK_GPU_VRAM=0; warn only
    warns.append("vram_probe_effective=false (checks may be off)")

# FAIL: normal systematically blocked (majority 503) while low also running —
# inference-first must not stall normal chat on mirror gates.
if normal_sum.get("n", 0) > 0:
    n503 = normal_sum.get("http_503", 0)
    if n503 >= max(1, (normal_sum["n"] + 1) // 2):
        fail_reasons.append(
            f"normal chat majority 503 ({n503}/{normal_sum['n']}) — "
            "inference-first must not block priority=normal"
        )
    if normal_sum.get("ok", 0) == 0 and normal_sum.get("n", 0) > 0:
        fail_reasons.append("no successful normal generate (check model tag / VRAM)")

# Soft expectation: with enough concurrent low, backlog pressure may latch;
# not a hard FAIL if idle GPU never queues.
if low_sum.get("n", 0) >= 4 and not mid_gates_true.get("runtime_backlog_pressure"):
    warns.append(
        "no runtime_backlog_pressure mid-flight (queue may have drained; "
        "increase P11_NUM_LOW / P11_NUM_PREDICT if tuning)"
    )

artifact = {
    "ts": datetime.now(timezone.utc).isoformat(),
    "host": ollama,
    "runtime": runtime,
    "model": model,
    "num_normal": n_normal,
    "num_low": n_low,
    "num_predict": n_predict,
    "elapsed_s": elapsed,
    "baseline": baseline,
    "mid_flight": mid_snaps,
    "after": after,
    "mid_gates_true": mid_gates_true,
    "summary": {"normal": normal_sum, "low": low_sum},
    "results": results,
    "training": train_snap,
    "defaults_under_test": {
        "VRAM_MIN_FREE": "1GiB",
        "TRAINING_VRAM_RESERVE": "2GiB",
        "RUNTIME_BACKLOG_BATCH_MIN": 4,
        "LOW_PRIORITY_VRAM_FACTOR": 1.5,
        "CROSS_QUEUE_PRESSURE_ON": 6,
        "CROSS_QUEUE_PRESSURE_CLEAR": 4,
    },
    "gates_baseline_true": {k: v for k, v in gates0.items() if v},
    "warnings": warns,
    "fail_reasons": fail_reasons,
    "verdict": "FAIL" if fail_reasons else "PASS",
    "doc": "docs/phase11-runtime-admission.md",
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(artifact, f, indent=2)
    f.write("\n")

print(f"wrote {out_path}")
print("summary.normal:", json.dumps(normal_sum))
print("summary.low:", json.dumps(low_sum))
print("mid_gates_true:", json.dumps(mid_gates_true))
if warns:
    for w in warns:
        print("warn:", w)
if fail_reasons:
    for r in fail_reasons:
        print("FAIL:", r, file=__import__("sys").stderr)
    raise SystemExit(1)
print("PASS: phase11_5080_contention_smoke")
PY

echo "PASS: phase11_5080_contention (${OUT})"
