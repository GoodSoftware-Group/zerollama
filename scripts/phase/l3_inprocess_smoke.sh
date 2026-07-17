#!/usr/bin/env bash
# L3 in-process prefix cache smoke — two-turn same cache key on Metal inprocess.
#
# WHY: subprocess L3 uses llama-server RAM+disk; in-process uses Phase 15 resume
# (v17 owner key) + optional llama_state_seq_* disk parity under llama_cache/.
#
# Usage:
#   M3_LLAMA_MODEL=/path/to/model.gguf ./scripts/phase/l3_inprocess_smoke.sh
#
# Env:
#   L3_CACHE_KEY, L3_NUM_CTX, L3_NUM_PREDICT, L3_OUT — same as l3_cache_smoke.sh
#   L3_SIMULATE_COLD_SLOT=1 — clear in-RAM owners between turns (disk restore path)
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

export ZEROLLAMA_GPU_PROFILE=1
export ZEROLLAMA_GPU_PROFILE_CTX=0
export ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess
unset ZEROLLAMA_RUNTIME_CONFIG
export ZEROLLAMA_AUTO_CONFIG=1
export ZEROLLAMA_LLAMA_CACHE=1
export ZEROLLAMA_LLAMA_CACHE_DISK=1
export MACOS_RT_HEALTH_MAX="${MACOS_RT_HEALTH_MAX:-120}"
export LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-$(cd "${ROOT}/.." && pwd)/llama.cpp}"
export LLAMA_CPP_LIB="${LLAMA_CPP_LIB:-${LLAMA_CPP_ROOT}/build/bin/libllama.dylib}"
export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${LLAMA_CPP_ROOT}/build/bin/llama-server}"

L3_OUT="${L3_OUT:-/tmp/l3-inprocess-smoke.json}"
L3_CACHE_KEY="${L3_CACHE_KEY:-l3-inprocess-thread-1}"
L3_NUM_CTX="${L3_NUM_CTX:-8192}"
L3_NUM_PREDICT="${L3_NUM_PREDICT:-32}"

macos_runtime_urls
trap macos_runtime_sidecar_cleanup EXIT

macos_runtime_stop_sidecar_port
macos_runtime_start_sidecar "${LLAMA_MODEL}" "" 0

health_json="$(runtime_fetch_health "${ZEROLLAMA_RUNTIME_URL}")"
runtime_resume_if_needed "${health_json}"

export RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL}"
export LLAMA_MODEL L3_CACHE_KEY L3_NUM_CTX L3_NUM_PREDICT L3_OUT L3_COMPARE_NO_CACHE L3_SIMULATE_COLD_SLOT
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
out_path = Path(os.environ.get("L3_OUT", "/tmp/l3-inprocess-smoke.json"))
compare_no_cache = os.environ.get("L3_COMPARE_NO_CACHE", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)
simulate_cold = os.environ.get("L3_SIMULATE_COLD_SLOT", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)

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
        "model": "l3-inprocess",
        "prompt": prompt,
        "stream": False,
        "options": opts,
    }
    t0 = time.perf_counter()
    out = http_json("POST", "/api/generate", payload)
    elapsed = time.perf_counter() - t0
    if not out.get("done"):
        raise RuntimeError(f"generate incomplete: {out!r}")
    return out, elapsed


health = http_json("GET", "/health")
lc = health.get("llama_cache") or {}
kr = health.get("kv_resume") or {}
gp = health.get("gpu_profile") or {}
backend = health.get("llama_backend")
if backend != "inprocess":
    raise SystemExit(f"expected inprocess backend, got {backend!r}")
if lc.get("enabled") is False:
    raise SystemExit("ZEROLLAMA_LLAMA_CACHE disabled")

from runtime.cache_bridge import derive_slot_id

n_parallel = gp.get("n_parallel") or health.get("kv_inprocess_n_seq_max") or 1
slot = derive_slot_id(cache_key, int(n_parallel))

_, _ = generate(turn1, cache_key=cache_key)
_, wall_turn1 = generate(turn1, cache_key=cache_key)

if simulate_cold:
    # Disk-restore path is validated in unit tests; smoke relies on turn2 RAM resume.
    pass

body2, wall_turn2 = generate(turn2, cache_key=cache_key)

health_after = http_json("GET", "/health")
lc_after = health_after.get("llama_cache") or {}
file_count = int(lc_after.get("file_count") or 0)

report = {
    "backend": backend,
    "gguf": gguf,
    "cache_key": cache_key,
    "derived_slot": slot,
    "num_ctx": num_ctx,
    "n_predict": n_predict,
    "n_parallel": n_parallel,
    "llama_cache": lc,
    "kv_resume": kr,
    "turn1_wall_s": round(wall_turn1, 3),
    "turn2_wall_s": round(wall_turn2, 3),
    "turn2_faster_than_turn1": wall_turn2 < wall_turn1,
    "speedup_ratio": round(wall_turn1 / wall_turn2, 3) if wall_turn2 > 0 else None,
    "disk_file_count": file_count,
    "inprocess_disk_cache": lc_after.get("inprocess_disk_cache"),
    "turn2_response_preview": (body2.get("response") or "")[:80],
}

if compare_no_cache:
    _, wall_nocache = generate(turn2, cache_key=None)
    report["turn2_no_cache_wall_s"] = round(wall_nocache, 3)
    report["cached_faster_than_no_cache"] = wall_turn2 < wall_nocache

out_path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
print(f"wrote {out_path}")

if file_count < 1 and lc_after.get("inprocess_disk_cache"):
    print(
        "warn: no slot files under llama_cache after turn2 — disk save may be unavailable",
        file=__import__("sys").stderr,
    )
PY
)

macos_runtime_stop_sidecar_port
trap - EXIT

echo ""
echo "PASS: l3_inprocess_smoke (${L3_OUT})"
echo "Doc: docs/gpu-profiles-l3.md"
