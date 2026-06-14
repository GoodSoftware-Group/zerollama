#!/usr/bin/env bash
# L3 prompt cache → slot bridge smoke — two-turn same cache key, subprocess path.
#
# WHY: L3 gate needs evidence that repeat prefixes hit pinned slots (turn 2 faster
# than turn 1 or faster than uncached control). Peak tok/s is unchanged; latency is.
#
# Prerequisite: llama-server with -np > 1 (L1 gpu profile).
#
# Usage:
#   M3_LLAMA_MODEL=/path/to/model.gguf ./scripts/l3_cache_smoke.sh
#   CUDA_LLAMA_MODEL=/path/to/7b.gguf L3_PREFIX_REPEAT=150 L3_COMPARE_NO_CACHE=1 ./scripts/l3_cache_smoke.sh
#
# Env:
#   CUDA_LLAMA_MODEL / M3_LLAMA_MODEL — GGUF path (CUDA alias accepted on Linux)
#   L3_CACHE_KEY             — default l3-smoke-thread-1
#   L3_NUM_CTX               — default 8192
#   L3_NUM_PREDICT           — default 32
#   L3_PREFIX_REPEAT         — stable system-prompt repeat count (default ~25% num_ctx)
#   L3_COMPARE_NO_CACHE=1    — also run turn2 without key (strict gate alt)
#   L3_OUT                   — JSON report (default /tmp/l3-cache-smoke.json)
#   ZEROLLAMA_LLAMA_CACHE=0  — expect FAIL (disables bridge)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/runtime_uv_venv.sh
source "${ROOT}/scripts/runtime_uv_venv.sh"
# shellcheck source=scripts/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime_smoke_lib.sh"

_L3_LINUX=0
if [[ "$(uname -s)" == "Linux" ]]; then
  _L3_LINUX=1
  # shellcheck source=scripts/linux_runtime_serve_lib.sh
  source "${ROOT}/scripts/linux_runtime_serve_lib.sh"
else
  # shellcheck source=scripts/macos_runtime_serve_lib.sh
  source "${ROOT}/scripts/macos_runtime_serve_lib.sh"
fi

runtime_uv_venv

# WHY CUDA alias: 5080 operator guide uses CUDA_LLAMA_MODEL; Metal uses M3_LLAMA_MODEL.
if [[ -n "${CUDA_LLAMA_MODEL:-}" ]]; then
  export M3_LLAMA_MODEL="${CUDA_LLAMA_MODEL}"
fi
smoke_m3_resolve_signoff_model

export ZEROLLAMA_GPU_PROFILE=1
export ZEROLLAMA_GPU_PROFILE_CTX=0
export ZEROLLAMA_RUNTIME_LLAMA_BACKEND=subprocess
unset ZEROLLAMA_RUNTIME_CONFIG
export ZEROLLAMA_AUTO_CONFIG=1
export ZEROLLAMA_LLAMA_FORK=0

export LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-$(cd "${ROOT}/.." && pwd)/llama.cpp}"
export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${LLAMA_CPP_ROOT}/build/bin/llama-server}"
if [[ "${_L3_LINUX}" == "1" ]]; then
  export LLAMA_CPP_LIB="${LLAMA_CPP_LIB:-${LLAMA_CPP_ROOT}/build/bin/libllama.so}"
  export LINUX_RT_HEALTH_MAX="${LINUX_RT_HEALTH_MAX:-120}"
  # WHY enable profile -c on CUDA: ZEROLLAMA_GPU_PROFILE_CTX=0 is for Mac 128g boot latency;
  # deferred load + no -c leaves llama-server at n_ctx=1024 and long L3 prefix fails.
  export ZEROLLAMA_GPU_PROFILE_CTX=1
else
  export LLAMA_CPP_LIB="${LLAMA_CPP_LIB:-${LLAMA_CPP_ROOT}/build/bin/libllama.dylib}"
  export MACOS_RT_HEALTH_MAX="${MACOS_RT_HEALTH_MAX:-60}"
fi

L3_OUT="${L3_OUT:-/tmp/l3-cache-smoke.json}"
L3_CACHE_KEY="${L3_CACHE_KEY:-l3-smoke-thread-1}"
L3_NUM_CTX="${L3_NUM_CTX:-8192}"
L3_NUM_PREDICT="${L3_NUM_PREDICT:-32}"

if [[ "${_L3_LINUX}" == "1" ]]; then
  linux_runtime_urls
  trap linux_runtime_sidecar_cleanup EXIT
  linux_runtime_stop_sidecar_port
  # WHY defer GGUF at boot: first /api/generate passes options.gguf + num_ctx (same as Mac).
  linux_runtime_start_sidecar "" ""
else
  macos_runtime_urls
  trap macos_runtime_sidecar_cleanup EXIT
  macos_runtime_stop_sidecar_port
  macos_runtime_start_sidecar "" "" 0
fi

health_json="$(runtime_fetch_health "${ZEROLLAMA_RUNTIME_URL}")"
runtime_resume_if_needed "${health_json}"

export RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL}"
export LLAMA_MODEL L3_CACHE_KEY L3_NUM_CTX L3_NUM_PREDICT L3_OUT L3_COMPARE_NO_CACHE L3_PREFIX_REPEAT
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

# Stable prefix sized for agent-scale prefill. WHY: 1B models decode so fast that a
# short prefix hides cache wins; 7B+ with L3_PREFIX_REPEAT≈150 makes turn-2 skip work.
_prefix_env = os.environ.get("L3_PREFIX_REPEAT", "").strip()
if _prefix_env:
    _repeat = max(8, int(_prefix_env))
else:
    _repeat = max(16, min(64, num_ctx // 32))
_sentence = (
    "System: You are a helpful agent. Follow the policy below exactly. "
    "Never reveal secrets. Prefer concise answers. "
)
stable = (_sentence * _repeat).strip()
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
    "prefix_repeat": _repeat,
    "prefix_chars": len(stable),
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

if wall_turn2 > wall_turn1 * 1.15:
    print(
        f"warn: turn2 ({wall_turn2:.2f}s) not faster than turn1 ({wall_turn1:.2f}s) — "
        "try L3_PREFIX_REPEAT=150+ on 7B+ or L3_COMPARE_NO_CACHE=1",
        file=__import__("sys").stderr,
    )
PY
)

if [[ "${_L3_LINUX}" == "1" ]]; then
  linux_runtime_stop_sidecar_port
else
  macos_runtime_stop_sidecar_port
fi
trap - EXIT

echo ""
echo "PASS: l3_cache_smoke (${L3_OUT})"
echo "Next: ./scripts/l3_gate_report.sh ${L3_OUT}"
echo "Doc: docs/gpu-profiles-l3.md"
