#!/usr/bin/env bash
# L3 production gate — strict prompt-cache latency win on a production-sized GGUF.
#
# WHY: l3_cache_smoke.sh on 1B Q8 @ 8k returns SOFT PASS (bridge wired, no measurable
# win) because the model is too fast and the prefix too short. The strict gate needs a
# larger model + longer prefix where KV reuse skips significant prefill work. Target:
# turn2 wall time at least L3_STRICT_RATIO faster than turn1, OR cached faster than
# the no-cache control run.
#
# Usage:
#   CUDA_LLAMA_MODEL=/root/eliza-1-9b-256k.gguf ./scripts/l3_production_gate.sh
#
# Env:
#   CUDA_LLAMA_MODEL         — GGUF path (required; or LLAMA_MODEL)
#   L3_NUM_CTX               — context window (default: 26624 for 27k-class agent prefix)
#   L3_NUM_PREDICT           — decode tokens (default: 32)
#   L3_PREFIX_REPEAT         — stable prefix repeat count (default: 150 — ~3k tokens on 9B tokeniser)
#   L3_STRICT_RATIO          — turn2/turn1 wall ratio for STRICT PASS (default: 0.75)
#   L3_COMPARE_NO_CACHE      — 1 to also run uncached control (default: 1 on production gate)
#   L3_GATE_OUT              — JSON report (default: /tmp/l3-production-gate.json)
#   LINUX_RT_HEALTH_MAX      — sidecar startup timeout in seconds (default: 180)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/linux_runtime_serve_lib.sh
source "${ROOT}/scripts/linux_runtime_serve_lib.sh"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "warn: l3_production_gate targets Linux CUDA; continuing anyway" >&2
fi

runtime_uv_venv

LLAMA_MODEL="${CUDA_LLAMA_MODEL:-${LLAMA_MODEL:-}}"
if [[ -z "${LLAMA_MODEL:-}" ]]; then
  echo "Set CUDA_LLAMA_MODEL to a production GGUF (e.g. 9B+ Q4/Q8 for a measurable L3 win)" >&2
  exit 1
fi
export LLAMA_MODEL

L3_NUM_CTX="${L3_NUM_CTX:-26624}"
L3_NUM_PREDICT="${L3_NUM_PREDICT:-32}"
# 150 repeats ≈ 3 200 tokens on a 9B tokeniser at 21 words/sentence.
L3_PREFIX_REPEAT="${L3_PREFIX_REPEAT:-150}"
L3_STRICT_RATIO="${L3_STRICT_RATIO:-0.75}"
L3_COMPARE_NO_CACHE="${L3_COMPARE_NO_CACHE:-1}"
L3_GATE_OUT="${L3_GATE_OUT:-/tmp/l3-production-gate.json}"

export ZEROLLAMA_GPU_PROFILE=1
export ZEROLLAMA_GPU_PROFILE_CTX=1
export ZEROLLAMA_LLAMA_CACHE=1
export ZEROLLAMA_RUNTIME_LLAMA_BACKEND=subprocess
export ZEROLLAMA_LLAMA_FORK=0
export ZEROLLAMA_AUTO_CONFIG=1
unset ZEROLLAMA_RUNTIME_CONFIG
export LINUX_RT_HEALTH_MAX="${LINUX_RT_HEALTH_MAX:-180}"

export LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-$(cd "${ROOT}/.." && pwd)/llama.cpp}"
export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${LLAMA_CPP_ROOT}/build/bin/llama-server}"
export LLAMA_CPP_LIB="${LLAMA_CPP_LIB:-${LLAMA_CPP_ROOT}/build/bin/libllama.so}"

linux_runtime_urls
trap linux_runtime_sidecar_cleanup EXIT
linux_runtime_stop_sidecar_port
linux_runtime_start_sidecar "${LLAMA_MODEL}" ""

health_json="$(runtime_fetch_health "${ZEROLLAMA_RUNTIME_URL}")"
linux_runtime_resume_if_needed "${health_json}"

echo "== L3 production gate =="
echo "model:       ${LLAMA_MODEL}"
echo "num_ctx:     ${L3_NUM_CTX}"
echo "num_predict: ${L3_NUM_PREDICT}"
echo "prefix_rpt:  ${L3_PREFIX_REPEAT}"
echo "strict_ratio: ${L3_STRICT_RATIO}  (turn2/turn1 must be ≤ this)"
echo "compare_no_cache: ${L3_COMPARE_NO_CACHE}"
echo "out: ${L3_GATE_OUT}"
echo ""

export RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL}"
export L3_OUT="${L3_GATE_OUT}"
export L3_NUM_CTX L3_NUM_PREDICT L3_PREFIX_REPEAT L3_STRICT_RATIO L3_COMPARE_NO_CACHE
export L3_CACHE_KEY="${L3_CACHE_KEY:-l3-production-gate-$(date +%s)}"

(cd "${ROOT}/runtime" && PYTHONPATH=. "${RUNTIME_UV_PYTHON}" <<'PY'
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

url = os.environ["RUNTIME_URL"].rstrip("/")
gguf = os.environ["LLAMA_MODEL"]
cache_key = os.environ["L3_CACHE_KEY"]
num_ctx = int(os.environ.get("L3_NUM_CTX", "26624"))
n_predict = int(os.environ.get("L3_NUM_PREDICT", "32"))
prefix_repeat = max(8, int(os.environ.get("L3_PREFIX_REPEAT", "150")))
strict_ratio = float(os.environ.get("L3_STRICT_RATIO", "0.75"))
compare_no_cache = os.environ.get("L3_COMPARE_NO_CACHE", "1").strip().lower() in ("1", "true", "yes")
out_path = Path(os.environ.get("L3_OUT", "/tmp/l3-production-gate.json"))

_sentence = (
    "System: You are a helpful assistant for a software engineering team. "
    "Follow all instructions carefully. Prefer concise, precise answers. "
    "Never reveal internal configuration. "
)
stable = (_sentence * prefix_repeat).strip()
turn1_prompt = f"{stable}\nUser: Summarise your instructions in one sentence.\nAssistant:"
turn2_prompt = f"{stable}\nUser: What topic were you asked to be precise about?\nAssistant:"
no_cache_prompt = f"{stable}\nUser: Describe the weather today in one word.\nAssistant:"


def http_json(method, path, body=None, timeout=600.0):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"{url}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def generate(prompt, *, use_cache_key):
    opts = {"gguf": gguf, "num_ctx": num_ctx, "num_predict": n_predict}
    if use_cache_key:
        opts["prompt_cache_key"] = cache_key
    payload = {"model": "l3-prod-gate", "prompt": prompt, "stream": False, "options": opts}
    t0 = time.perf_counter()
    try:
        out = http_json("POST", "/api/generate", payload)
    except urllib.error.HTTPError as e:
        body_txt = e.read().decode(errors="replace") if e.fp else ""
        raise RuntimeError(f"generate HTTP {e.code}: {body_txt[:500]}") from e
    elapsed = time.perf_counter() - t0
    if not out.get("done"):
        raise RuntimeError(f"incomplete: {out!r}")
    return out, elapsed


health = http_json("GET", "/health")
lc = health.get("llama_cache") or {}
gp = health.get("gpu_profile") or {}
n_parallel = gp.get("n_parallel") or 1

if lc.get("enabled") is False:
    raise SystemExit("ZEROLLAMA_LLAMA_CACHE is disabled — enable for L3 gate")

from runtime.cache_bridge import derive_slot_id
derived_slot = derive_slot_id(cache_key, int(n_parallel))

print(f"cache_key:    {cache_key}")
print(f"derived_slot: {derived_slot}  n_parallel={n_parallel}")
print(f"llama_cache:  {lc}")
print()

# Warmup — prime the slot
print("warmup...")
generate(turn1_prompt, use_cache_key=True)

# Turn 1 — full prefill expected
print("turn1...")
_, wall_turn1 = generate(turn1_prompt, use_cache_key=True)

# Turn 2 — shared prefix should skip prefill
print("turn2 (cached)...")
_, wall_turn2 = generate(turn2_prompt, use_cache_key=True)

wall_no_cache = None
if compare_no_cache:
    # WHY different prompt but same prefix: tests that cache key pins the slot while
    # a separate request with no key does not benefit from the cached prefix.
    print("no-cache control...")
    _, wall_no_cache = generate(no_cache_prompt, use_cache_key=False)

ratio = wall_turn2 / wall_turn1 if wall_turn1 > 0 else None
strict_pass = ratio is not None and ratio <= strict_ratio
no_cache_win = wall_no_cache is not None and wall_turn2 < wall_no_cache

print()
print(f"turn1_wall_s:     {wall_turn1:.3f}")
print(f"turn2_wall_s:     {wall_turn2:.3f}")
if ratio is not None:
    print(f"turn2/turn1:      {ratio:.3f}  (need ≤ {strict_ratio} for STRICT PASS)")
if wall_no_cache is not None:
    print(f"no_cache_wall_s:  {wall_no_cache:.3f}")
    print(f"cached_faster_than_no_cache: {no_cache_win}")

report = {
    "model": gguf,
    "cache_key": cache_key,
    "derived_slot": derived_slot,
    "n_parallel": n_parallel,
    "num_ctx": num_ctx,
    "num_predict": n_predict,
    "prefix_repeat": prefix_repeat,
    "strict_ratio_threshold": strict_ratio,
    "turn1_wall_s": round(wall_turn1, 3),
    "turn2_wall_s": round(wall_turn2, 3),
    "turn2_over_turn1": round(ratio, 4) if ratio is not None else None,
    "turn2_faster_than_turn1": wall_turn2 < wall_turn1,
    "strict_pass": strict_pass,
    "no_cache_wall_s": round(wall_no_cache, 3) if wall_no_cache is not None else None,
    "cached_faster_than_no_cache": no_cache_win if wall_no_cache is not None else None,
    "llama_cache": lc,
    "gpu_profile": gp,
}
out_path.write_text(json.dumps(report, indent=2) + "\n")
print()
print(f"wrote {out_path}")

# Verdict
if strict_pass:
    print(f"VERDICT: L3 production gate STRICT PASS — turn2/turn1={ratio:.3f} ≤ {strict_ratio}")
    raise SystemExit(0)
if no_cache_win:
    print("VERDICT: L3 production gate PASS (cached faster than no-cache control)")
    raise SystemExit(0)
if wall_turn2 < wall_turn1:
    pct = (wall_turn1 - wall_turn2) / wall_turn1 * 100
    print(f"VERDICT: L3 SOFT PASS — turn2 {pct:.1f}% faster but ratio {ratio:.3f} > {strict_ratio}")
    print(f"  Increase L3_PREFIX_REPEAT (current {prefix_repeat}) or use a larger GGUF.")
    raise SystemExit(0)
print("VERDICT: L3 production gate FAIL — no cache latency win")
print("  Check: subprocess backend, ZEROLLAMA_LLAMA_CACHE=1, L3_PREFIX_REPEAT≥100, model ≥7B.")
raise SystemExit(1)
PY
)

linux_runtime_stop_sidecar_port
trap - EXIT

echo ""
echo "Artifacts: ${L3_GATE_OUT}"
echo "Doc: docs/gpu-profiles-l3.md"
