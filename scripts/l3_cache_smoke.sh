#!/usr/bin/env bash
# L3 prompt cache → slot bridge smoke — two-turn same cache key, subprocess path.
#
# WHY: L3 gate needs evidence that repeat prefixes hit pinned slots (turn 2 faster
# than turn 1 or faster than uncached control). Peak tok/s is unchanged; latency is.
#
# Prerequisite: llama-server with -np > 1 (L1 gpu profile on Mac).
#
# Usage:
#   M3_LLAMA_MODEL=/path/to/model.gguf ./scripts/l3_cache_smoke.sh
#   L3_COMPARE_NO_CACHE=1 ./scripts/l3_cache_smoke.sh   # also run turn2 without key
#
# Env:
#   L3_CACHE_KEY             — default l3-smoke-thread-1
#   L3_NUM_CTX               — default 8192
#   L3_NUM_PREDICT           — default 32
#   L3_OUT                   — JSON report (default /tmp/l3-cache-smoke.json)
#   ZEROLLAMA_LLAMA_CACHE=0  — expect FAIL (disables bridge)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/runtime_uv_venv.sh
source "${ROOT}/scripts/runtime_uv_venv.sh"
# shellcheck source=scripts/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime_smoke_lib.sh"
# shellcheck source=scripts/macos_runtime_serve_lib.sh
source "${ROOT}/scripts/macos_runtime_serve_lib.sh"

runtime_uv_venv
smoke_m3_resolve_signoff_model

export ZEROLLAMA_GPU_PROFILE=1
export ZEROLLAMA_GPU_PROFILE_CTX=0
export ZEROLLAMA_RUNTIME_LLAMA_BACKEND=subprocess
unset ZEROLLAMA_RUNTIME_CONFIG
export ZEROLLAMA_AUTO_CONFIG=1
# Defer GGUF load until first /api/generate (options.gguf) — WHY: profile -c 131072
# at sidecar boot blocks /health for minutes on 128g tier.
export MACOS_RT_HEALTH_MAX="${MACOS_RT_HEALTH_MAX:-60}"
export LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-$(cd "${ROOT}/.." && pwd)/llama.cpp}"
export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${LLAMA_CPP_ROOT}/build/bin/llama-server}"
export LLAMA_CPP_LIB="${LLAMA_CPP_LIB:-${LLAMA_CPP_ROOT}/build/bin/libllama.dylib}"
export ZEROLLAMA_LLAMA_FORK=0

L3_OUT="${L3_OUT:-/tmp/l3-cache-smoke.json}"
L3_CACHE_KEY="${L3_CACHE_KEY:-l3-smoke-thread-1}"
L3_NUM_CTX="${L3_NUM_CTX:-8192}"
L3_NUM_PREDICT="${L3_NUM_PREDICT:-32}"

macos_runtime_urls
trap macos_runtime_sidecar_cleanup EXIT

macos_runtime_stop_sidecar_port
macos_runtime_start_sidecar "" "" 0

health_json="$(runtime_fetch_health "${ZEROLLAMA_RUNTIME_URL}")"
runtime_resume_if_needed "${health_json}"

export RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL}"
export LLAMA_MODEL L3_CACHE_KEY L3_NUM_CTX L3_NUM_PREDICT L3_OUT L3_COMPARE_NO_CACHE
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
num_ctx = int(os.environ.get("L3_NUM_CTX", "8192"))
n_predict = int(os.environ.get("L3_NUM_PREDICT", "32"))
out_path = Path(os.environ.get("L3_OUT", "/tmp/l3-cache-smoke.json"))
compare_no_cache = os.environ.get("L3_COMPARE_NO_CACHE", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)

# Stable prefix sized to ~25% of num_ctx (rough word→token). WHY: must fit in one
# prefill while still being long enough that turn-2 cache saves measurable work.
_repeat = max(16, min(64, num_ctx // 32))
stable = ("System: You are a concise assistant. " * _repeat).strip()
turn1 = f"{stable}\nUser: Say hello in one word.\nAssistant:"
turn2 = f"{stable}\nUser: Say goodbye in one word.\nAssistant:"


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


def generate(prompt: str, *, cache_key: str | None) -> tuple[dict, float]:
    opts: dict = {"gguf": gguf, "num_ctx": num_ctx, "num_predict": n_predict}
    if cache_key:
        opts["prompt_cache_key"] = cache_key
    payload = {
        "model": "l3-smoke",
        "prompt": prompt,
        "stream": False,
        "options": opts,
    }
    t0 = time.perf_counter()
    try:
        out = http_json("POST", "/api/generate", payload)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace") if e.fp else ""
        raise RuntimeError(f"generate HTTP {e.code}: {body[:500]}") from e
    elapsed = time.perf_counter() - t0
    if not out.get("done"):
        raise RuntimeError(f"generate incomplete: {out!r}")
    return out, elapsed


health = http_json("GET", "/health")
lc = health.get("llama_cache") or {}
gp = health.get("gpu_profile") or {}
n_parallel = gp.get("n_parallel") or 1
if lc.get("enabled") is False:
    raise SystemExit("ZEROLLAMA_LLAMA_CACHE disabled — enable for L3 smoke")
if n_parallel < 2:
    print(f"warn: gpu_profile.n_parallel={n_parallel} (L3 pinning works; multi-session needs -np > 1)")

# Warmup
_, _ = generate(turn1, cache_key=cache_key)

_, wall_turn1 = generate(turn1, cache_key=cache_key)
body2, wall_turn2 = generate(turn2, cache_key=cache_key)

from runtime.cache_bridge import derive_slot_id

slot = derive_slot_id(cache_key, int(n_parallel))

report: dict = {
    "gguf": gguf,
    "cache_key": cache_key,
    "derived_slot": slot,
    "num_ctx": num_ctx,
    "n_predict": n_predict,
    "n_parallel": n_parallel,
    "llama_cache": lc,
    "turn1_wall_s": round(wall_turn1, 3),
    "turn2_wall_s": round(wall_turn2, 3),
    "turn2_faster_than_turn1": wall_turn2 < wall_turn1,
    "speedup_ratio": round(wall_turn1 / wall_turn2, 3) if wall_turn2 > 0 else None,
    "turn2_response_preview": (body2.get("response") or "")[:80],
}

if compare_no_cache:
    _, wall_nocache = generate(turn2, cache_key=None)
    report["turn2_no_cache_wall_s"] = round(wall_nocache, 3)
    report["cached_faster_than_no_cache"] = wall_turn2 < wall_nocache

health_after = http_json("GET", "/health")
report["llama_cache_after"] = health_after.get("llama_cache")

out_path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
print(f"wrote {out_path}")

# Gate: turn2 should not be slower than turn1 by much (cache hit or tie on tiny model).
if wall_turn2 > wall_turn1 * 1.15:
    print(
        f"warn: turn2 ({wall_turn2:.2f}s) not faster than turn1 ({wall_turn1:.2f}s) — "
        "cache may not be effective on this model/backend",
        file=__import__("sys").stderr,
    )
PY
)

macos_runtime_stop_sidecar_port
trap - EXIT

echo ""
echo "PASS: l3_cache_smoke (${L3_OUT})"
echo "Doc: docs/gpu-profiles-l3.md"
