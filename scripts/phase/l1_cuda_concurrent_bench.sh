#!/usr/bin/env bash
# L1 CUDA concurrent bench — aggregate tok/s and TTFT under N parallel requests.
#
# WHY: l1_cuda_calibrate.sh validates single-stream. The L1 profile sets n_parallel=2
# to amortise prefill across concurrent sessions. This script fires N simultaneous
# /api/generate requests and measures:
#   - aggregate tok/s (sum of per-request tok/s across all threads)
#   - TTFT / total wall per thread
#   - profile OFF vs ON A/B
#
# Usage:
#   CUDA_LLAMA_MODEL=/root/eliza-1-9b-256k.gguf ./scripts/phase/l1_cuda_concurrent_bench.sh
#   L1C_N=4 L1C_SWEEP_NP="1,2,4" ./scripts/phase/l1_cuda_concurrent_bench.sh
#
# Env:
#   CUDA_LLAMA_MODEL         — GGUF path (required; or LLAMA_MODEL)
#   L1C_N                    — concurrent request count (default: 2 — matches n_parallel)
#   L1C_NUM_CTX              — context window per request (default: 8192)
#   L1C_NUM_PREDICT          — decode tokens per request (default: 128)
#   L1C_BENCH_RUNS           — timed rounds (default: 2)
#   L1C_SWEEP_NP             — comma list of n_parallel overrides to sweep (profile ON legs)
#   L1C_SKIP_OFF=1           — skip profile-off baseline
#   L1C_OUT_DIR              — default /tmp/l1-cuda-concurrent
#   L1C_ENFORCE=1            — exit 1 when best ON concurrent leg does not beat OFF
#   LINUX_RT_HEALTH_MAX      — sidecar startup timeout in seconds (default: 120)
#
# OFF leg note: profile OFF uses n_parallel=1 — expect 1×502 Bad Gateway per run when
# L1C_N=2 (second slot rejected). Aggregate tok/s still sums successful threads; ON leg
# should show 0 errors when n_parallel matches L1C_N.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/runtime/linux_runtime_serve_lib.sh
source "${ROOT}/scripts/runtime/linux_runtime_serve_lib.sh"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "warn: l1_cuda_concurrent_bench targets Linux CUDA; continuing anyway" >&2
fi

runtime_uv_venv

LLAMA_MODEL="${CUDA_LLAMA_MODEL:-${LLAMA_MODEL:-}}"
if [[ -z "${LLAMA_MODEL:-}" ]]; then
  echo "Set CUDA_LLAMA_MODEL to a production GGUF (e.g. 7B–9B Q8 on 16GB)" >&2
  exit 1
fi
export LLAMA_MODEL

OUT_DIR="${L1C_OUT_DIR:-/tmp/l1-cuda-concurrent}"
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

# WHY: embedded zerollama on :8081 races uv sidecar (go-coordination + 502 on generate).
linux_runtime_stop_sidecar_port
fuser -k 8080/tcp 2>/dev/null || true
for _zpid in $(pgrep -x zerollama 2>/dev/null || true); do kill -9 "$_zpid" 2>/dev/null || true; done
sleep 2

L1C_N="${L1C_N:-2}"
L1C_NUM_CTX="${L1C_NUM_CTX:-8192}"
L1C_NUM_PREDICT="${L1C_NUM_PREDICT:-128}"
L1C_BENCH_RUNS="${L1C_BENCH_RUNS:-2}"

export ZEROLLAMA_RUNTIME_LLAMA_BACKEND=subprocess
export ZEROLLAMA_AUTO_CONFIG=1
export ZEROLLAMA_GPU_PROFILE_CTX=1
unset ZEROLLAMA_RUNTIME_CONFIG
export LINUX_RT_HEALTH_MAX="${LINUX_RT_HEALTH_MAX:-120}"

l1_export_llama_binary_env "${ROOT}"

linux_runtime_urls
trap linux_runtime_sidecar_cleanup EXIT

  _run_concurrent_leg() {
  local label="$1"
  local profile="$2"    # 0 | 1
  local extra_args="${3:-}"
  local out="${OUT_DIR}/${label}.json"

  echo ""
  echo "== L1 concurrent: ${label} (profile=${profile} n=${L1C_N} extra='${extra_args}') =="

  linux_runtime_stop_sidecar_port
  # WHY fork=0: L1 concurrent validates n_parallel/batch with stock q8_0 KV (L2 owns QJL/Polar).
  export ZEROLLAMA_LLAMA_FORK=0

  ZEROLLAMA_GPU_PROFILE="${profile}" \
  ZEROLLAMA_LLAMA_FORK=0 \
  LLAMA_SERVER_EXTRA_ARGS="${extra_args}" \
  linux_runtime_start_sidecar "${LLAMA_MODEL}" ""

  local health_json
  health_json="$(runtime_fetch_health "${ZEROLLAMA_RUNTIME_URL}")"
  linux_runtime_resume_if_needed "${health_json}"

  export RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL}"
  export L1C_LABEL="${label}"
  export L1C_PROFILE="${profile}"
  export L1C_EXTRA_ARGS="${extra_args}"

  (cd "${ROOT}/runtime" && PYTHONPATH=. "${RUNTIME_UV_PYTHON}" <<'PY'
import json
import os
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

url = os.environ["RUNTIME_URL"].rstrip("/")
gguf = os.environ["LLAMA_MODEL"]
n_concurrent = int(os.environ.get("L1C_N", "2"))
num_ctx = int(os.environ.get("L1C_NUM_CTX", "8192"))
n_predict = int(os.environ.get("L1C_NUM_PREDICT", "128"))
bench_runs = max(1, int(os.environ.get("L1C_BENCH_RUNS", "2")))
label = os.environ["L1C_LABEL"]
out_path = Path(os.environ.get("L1C_OUT_DIR", "/tmp/l1-cuda-concurrent")) / f"{label}.json"

# Distinct prompts per slot — different text avoids KV cache sharing between threads.
_PROMPTS = [
    "Describe the history of neural networks. Be detailed.\n1.",
    "Explain the steps of a transformer forward pass. Be detailed.\n1.",
    "List the main CUDA optimisation techniques for LLMs. Be detailed.\n1.",
    "Compare GGUF quantisation formats Q4_K and Q8_0. Be detailed.\n1.",
    "What are the benefits of continuous batching in LLM inference? Be detailed.\n1.",
    "Explain paged KV cache and why it reduces memory fragmentation. Be detailed.\n1.",
    "Describe the NVIDIA Blackwell architecture improvements over Hopper. Be detailed.\n1.",
    "What is speculative decoding and how does draft models work? Be detailed.\n1.",
]


def http_json(method: str, path: str, body: dict | None = None, timeout: float = 600.0):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"{url}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def generate_one(thread_idx: int, results: list, errors: list, barrier: threading.Barrier) -> None:
    prompt = _PROMPTS[thread_idx % len(_PROMPTS)]
    payload = {
        "model": f"l1c-t{thread_idx}",
        "prompt": prompt,
        "stream": False,
        "options": {"gguf": gguf, "num_ctx": num_ctx, "num_predict": n_predict},
    }
    barrier.wait()  # All threads start together
    t0 = time.perf_counter()
    try:
        out = http_json("POST", "/api/generate", payload, timeout=600.0)
        elapsed = time.perf_counter() - t0
        eval_count = out.get("eval_count") or n_predict
        eval_duration_ns = out.get("eval_duration") or 0
        tok_s = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 else (eval_count / max(elapsed, 1e-6))
        results[thread_idx] = {
            "thread": thread_idx,
            "wall_s": round(elapsed, 3),
            "eval_count": eval_count,
            "tok_s": round(tok_s, 2),
            "done": out.get("done", False),
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        errors[thread_idx] = str(e)
        results[thread_idx] = {"thread": thread_idx, "wall_s": round(elapsed, 3), "error": str(e)}


health = http_json("GET", "/health")
gp = health.get("gpu_profile") or {}

# Warmup — single request, no barrier.
try:
    http_json("POST", "/api/generate", {
        "model": "l1c-warmup",
        "prompt": "warmup",
        "stream": False,
        "options": {"gguf": gguf, "num_ctx": num_ctx, "num_predict": 4},
    })
except Exception:
    pass

round_results = []
for run_idx in range(bench_runs):
    results = [None] * n_concurrent
    errors = [None] * n_concurrent
    barrier = threading.Barrier(n_concurrent)
    threads = [
        threading.Thread(target=generate_one, args=(i, results, errors, barrier), daemon=True)
        for i in range(n_concurrent)
    ]
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=660)
    t_total = time.perf_counter() - t_start

    valid = [r for r in results if r and "tok_s" in r]
    agg_tok_s = sum(r["tok_s"] for r in valid)
    max_wall = max((r["wall_s"] for r in valid), default=0.0)
    round_results.append({
        "run": run_idx,
        "threads": results,
        "agg_tok_s": round(agg_tok_s, 2),
        "max_wall_s": round(max_wall, 3),
        "total_wall_s": round(t_total, 3),
        "error_count": sum(1 for e in errors if e),
    })
    print(f"  run {run_idx}: agg={agg_tok_s:.1f} tok/s  max_wall={max_wall:.2f}s  errors={sum(1 for e in errors if e)}")

agg_values = [r["agg_tok_s"] for r in round_results]
mean_agg = statistics.mean(agg_values) if agg_values else 0.0

report = {
    "label": label,
    "n_concurrent": n_concurrent,
    "num_ctx": num_ctx,
    "num_predict": n_predict,
    "bench_runs": bench_runs,
    "gpu_profile": gp,
    "rounds": round_results,
    "agg_tok_s_mean": round(mean_agg, 2),
    "agg_tok_s_runs": agg_values,
}
out_path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"agg_tok_s_mean": round(mean_agg, 2), "n_concurrent": n_concurrent, "gpu_profile_id": gp.get("id"), "n_parallel": gp.get("n_parallel")}))
PY
  ) | tee -a "${OUT_DIR}/${label}.log"
}

echo "== L1 CUDA concurrent bench =="
echo "model: ${LLAMA_MODEL}"
echo "n_concurrent=${L1C_N}  ctx=${L1C_NUM_CTX}  predict=${L1C_NUM_PREDICT}  runs=${L1C_BENCH_RUNS}"
echo "out: ${OUT_DIR}/"

if [[ "${L1C_SKIP_OFF:-0}" != "1" ]]; then
  _run_concurrent_leg "profile-off" "0" ""
fi

_run_concurrent_leg "profile-on-default" "1" ""

IFS=',' read -r -a _np_sweep <<< "${L1C_SWEEP_NP:-}"
for np in "${_np_sweep[@]}"; do
  np="${np// /}"
  [[ -z "${np}" ]] && continue
  _run_concurrent_leg "profile-on-np${np}" "1" "-np ${np}"
done

linux_runtime_stop_sidecar_port
trap - EXIT

echo ""
echo "== L1 concurrent bench summary =="
python3 <<PY
import json
import os
from pathlib import Path

out_dir = Path("${OUT_DIR}")
rows = []
for p in sorted(out_dir.glob("*.json")):
    d = json.loads(p.read_text())
    gp = d.get("gpu_profile") or {}
    rows.append({
        "label": d.get("label", p.stem),
        "agg_tok_s": d.get("agg_tok_s_mean", 0.0),
        "n_concurrent": d.get("n_concurrent"),
        "n_parallel": gp.get("n_parallel"),
        "gpu_profile_id": gp.get("id"),
    })

off = next((r for r in rows if r["label"] == "profile-off"), None)
n = rows[0]["n_concurrent"] if rows else "${L1C_N}"
print(f"n_concurrent={n}")
print(f"{'label':<26} {'agg tok/s':>10}  {'n_parallel':>10}  vs OFF")
for r in rows:
    delta = ""
    if off and off["agg_tok_s"] and r["agg_tok_s"]:
        pct = (r["agg_tok_s"] - off["agg_tok_s"]) / off["agg_tok_s"] * 100
        delta = f"{pct:+.1f}%"
    print(f"{r['label']:<26} {r['agg_tok_s']:>10.2f}  {str(r['n_parallel'] or '-'):>10}  {delta}")

if off:
    on_rows = [r for r in rows if r["label"] != "profile-off" and r["agg_tok_s"]]
    if on_rows:
        best = max(on_rows, key=lambda r: r["agg_tok_s"])
        pct = (best["agg_tok_s"] - off["agg_tok_s"]) / off["agg_tok_s"] * 100
        verdict = "PASS" if pct > 0 else ("FLAT" if pct == 0 else "REGRESS")
        print()
        print(f"VERDICT: concurrent {verdict} — best ON {best['agg_tok_s']} tok/s ({pct:+.1f}% vs OFF)")
        if pct <= 0 and os.environ.get("L1C_ENFORCE", "0") == "1":
            raise SystemExit(1)
        if pct < -5:
            print("  Recommendation: reduce n_parallel in rtx-5080.json or bump -b/-ub")
PY

echo ""
echo "Artifacts: ${OUT_DIR}/"
echo "Next: review summary above; update runtime/configs/gpu/rtx-5080.json if concurrent regresses."
echo "Doc: docs/gpu-profiles-l1.md"
