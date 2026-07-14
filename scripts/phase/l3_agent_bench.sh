#!/usr/bin/env bash
# L3 agent workload bench — multi-turn prefix reuse (in-process or subprocess).
#
# WHY: single turn-1 vs turn-2 smokes miss cumulative agent latency wins.
#
# Usage:
#   M3_LLAMA_MODEL=/path/to/model.gguf ./scripts/phase/l3_agent_bench.sh
#   L3_BACKEND=subprocess ./scripts/phase/l3_agent_bench.sh
#
# Env:
#   L3_BACKEND             — inprocess (default) or subprocess
#   L3_AGENT_TURNS         — default 5
#   L3_AGENT_OUT           — JSON report (default /tmp/l3-agent-bench.json)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/runtime/runtime_uv_venv.sh
source "${ROOT}/scripts/runtime/runtime_uv_venv.sh"
# shellcheck source=scripts/runtime/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime/runtime_smoke_lib.sh"
# shellcheck source=scripts/runtime/macos_runtime_serve_lib.sh
source "${ROOT}/scripts/runtime/macos_runtime_serve_lib.sh"

runtime_uv_venv
smoke_m3_resolve_signoff_model

L3_BACKEND="${L3_BACKEND:-inprocess}"
L3_AGENT_TURNS="${L3_AGENT_TURNS:-5}"
L3_AGENT_OUT="${L3_AGENT_OUT:-/tmp/l3-agent-bench.json}"
L3_CACHE_KEY="${L3_CACHE_KEY:-l3-agent-bench-$(date +%s)}"
L3_NUM_CTX="${L3_NUM_CTX:-8192}"
L3_NUM_PREDICT="${L3_NUM_PREDICT:-16}"

export ZEROLLAMA_GPU_PROFILE=1
export ZEROLLAMA_GPU_PROFILE_CTX=0
export ZEROLLAMA_LLAMA_CACHE=1
export ZEROLLAMA_AUTO_CONFIG=1
unset ZEROLLAMA_RUNTIME_CONFIG
export MACOS_RT_HEALTH_MAX="${MACOS_RT_HEALTH_MAX:-120}"
export LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-$(cd "${ROOT}/.." && pwd)/llama.cpp}"
export LLAMA_CPP_LIB="${LLAMA_CPP_LIB:-${LLAMA_CPP_ROOT}/build/bin/libllama.dylib}"
export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${LLAMA_CPP_ROOT}/build/bin/llama-server}"

if [[ "${L3_BACKEND}" == "inprocess" ]]; then
  export ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess
  export ZEROLLAMA_LLAMA_CACHE_DISK=1
else
  export ZEROLLAMA_RUNTIME_LLAMA_BACKEND=subprocess
  export ZEROLLAMA_LLAMA_FORK=0
fi

macos_runtime_urls
trap macos_runtime_sidecar_cleanup EXIT

macos_runtime_stop_sidecar_port
macos_runtime_start_sidecar "${LLAMA_MODEL}" "" 0
runtime_resume_if_needed "$(runtime_fetch_health "${ZEROLLAMA_RUNTIME_URL}")"

export RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL}"
export LLAMA_MODEL L3_CACHE_KEY L3_NUM_CTX L3_NUM_PREDICT L3_AGENT_TURNS L3_AGENT_OUT L3_BACKEND
(cd "${ROOT}/runtime" && PYTHONPATH=. "${RUNTIME_UV_PYTHON}" <<'PY'
import json
import os
import time
import urllib.request
from pathlib import Path

url = os.environ["RUNTIME_URL"].rstrip("/")
gguf = os.environ["LLAMA_MODEL"]
cache_key = os.environ["L3_CACHE_KEY"]
num_ctx = int(os.environ.get("L3_NUM_CTX", "8192"))
n_predict = int(os.environ.get("L3_NUM_PREDICT", "16"))
turns = max(2, int(os.environ.get("L3_AGENT_TURNS", "5")))
out_path = Path(os.environ.get("L3_AGENT_OUT", "/tmp/l3-agent-bench.json"))
backend = os.environ.get("L3_BACKEND", "inprocess")

_repeat = max(16, min(48, num_ctx // 40))
system = ("System: You are a helpful coding agent. Follow instructions precisely. " * _repeat).strip()


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


def generate(user_msg: str, *, cache_key: str | None) -> tuple[dict, float]:
    prompt = f"{system}\nUser: {user_msg}\nAssistant:"
    opts: dict = {"gguf": gguf, "num_ctx": num_ctx, "num_predict": n_predict}
    if cache_key:
        opts["prompt_cache_key"] = cache_key
    payload = {
        "model": "l3-agent",
        "prompt": prompt,
        "stream": False,
        "options": opts,
    }
    t0 = time.perf_counter()
    out = http_json("POST", "/api/generate", payload)
    elapsed = time.perf_counter() - t0
    if not out.get("done"):
        raise RuntimeError(out)
    return out, elapsed


health = http_json("GET", "/health")
walls_cached: list[float] = []
walls_cold: list[float] = []

generate("warmup", cache_key=cache_key)

for i in range(turns):
    _, w = generate(f"turn {i+1}: reply briefly", cache_key=cache_key)
    walls_cached.append(w)

for i in range(min(2, turns)):
    _, w = generate(f"cold {i+1}: reply briefly", cache_key=None)
    walls_cold.append(w)

health_after = http_json("GET", "/health")
lc = health_after.get("llama_cache") or {}

turn1 = walls_cached[0] if walls_cached else None
tail_mean = sum(walls_cached[1:]) / max(1, len(walls_cached) - 1) if len(walls_cached) > 1 else None
cold_mean = sum(walls_cold) / max(1, len(walls_cold)) if walls_cold else None

report = {
    "backend": backend,
    "cache_key": cache_key,
    "turns": turns,
    "num_ctx": num_ctx,
    "wall_s_cached": [round(x, 3) for x in walls_cached],
    "wall_s_cold": [round(x, 3) for x in walls_cold],
    "turn1_wall_s": round(turn1, 3) if turn1 is not None else None,
    "tail_mean_wall_s": round(tail_mean, 3) if tail_mean is not None else None,
    "cold_mean_wall_s": round(cold_mean, 3) if cold_mean is not None else None,
    "tail_faster_than_turn1": tail_mean is not None and turn1 is not None and tail_mean < turn1,
    "cached_faster_than_cold": tail_mean is not None and cold_mean is not None and tail_mean < cold_mean,
    "llama_cache": lc,
}

out_path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
print(f"wrote {out_path}")
PY
)

macos_runtime_stop_sidecar_port
trap - EXIT

echo ""
echo "PASS: l3_agent_bench (${L3_AGENT_OUT})"
