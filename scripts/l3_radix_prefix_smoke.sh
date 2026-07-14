#!/usr/bin/env bash
# Cross-slot Radix prefix share — offline gate (no GPU).
#
# Usage:
#   ./scripts/l3_radix_prefix_smoke.sh
# Live (needs local GGUF + patched vendor llama-server with POST /kv/seq-copy):
#   M3_LLAMA_MODEL=/path/model.gguf L3_RADIX_LIVE=1 ./scripts/l3_radix_prefix_smoke.sh
# 5080 / CUDA (CUDA_LLAMA_MODEL alias — same as l3_cache_smoke):
#   CUDA_LLAMA_MODEL=/root/eliza-1-9b-256k.gguf L3_RADIX_LIVE=1 ./scripts/l3_radix_prefix_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${ROOT}/runtime"
cd "${RUNTIME}"

OUT="${L3_RADIX_OUT:-/tmp/l3-radix-prefix-smoke.json}"

echo "== radix prefix share pytest =="
PYTHONPATH=. python3 -m pytest \
  tests/test_radix_prefix_share.py \
  tests/test_radix_seq_copy.py \
  -q

echo "== radix offline plan replay =="
PYTHONPATH=. python3 <<PY
import json
import os
from pathlib import Path

from runtime.kv.prefix_block_pool import (
    build_model_scope,
    get_prefix_block_pool,
    reset_prefix_block_pools_for_tests,
)
from runtime.kv.radix_prefix_share import find_radix_share_plan

os.environ["ZEROLLAMA_RADIX_PREFIX_SHARE"] = "1"
reset_prefix_block_pools_for_tests()

scope = build_model_scope(model_hash="radix-smoke")
shared = list(range(2000, 2000 + 1024))
pool = get_prefix_block_pool(model_scope=scope)
pool.register_prefix(
    shared,
    scope=scope,
    seq_pos=1024,
    session_key="donor-session",
    slot_id=2,
)

plan = find_radix_share_plan(
    shared,
    target_slot=7,
    model_hash="radix-smoke",
    seq_pos=0,
)
assert plan is not None
assert plan.source_slot == 2 and plan.copy_tokens == 1024

report = {
    "offline_replay": "ok",
    "source_slot": plan.source_slot,
    "target_slot": plan.target_slot,
    "copy_tokens": plan.copy_tokens,
}
Path("${OUT}").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY

if [[ "${L3_RADIX_LIVE:-0}" != "1" ]]; then
  echo "wrote ${OUT} (set L3_RADIX_LIVE=1 for two-key runtime smoke)"
  exit 0
fi

cd "${ROOT}"
# shellcheck source=scripts/runtime_uv_venv.sh
source "${ROOT}/scripts/runtime_uv_venv.sh"
# shellcheck source=scripts/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime_smoke_lib.sh"
if [[ "$(uname -s)" == "Linux" ]]; then
  # shellcheck source=scripts/linux_runtime_serve_lib.sh
  source "${ROOT}/scripts/linux_runtime_serve_lib.sh"
  linux_runtime_urls
  trap linux_runtime_sidecar_cleanup EXIT
  linux_runtime_stop_sidecar_port
else
  # shellcheck source=scripts/macos_runtime_serve_lib.sh
  source "${ROOT}/scripts/macos_runtime_serve_lib.sh"
  macos_runtime_urls
  trap macos_runtime_sidecar_cleanup EXIT
  macos_runtime_stop_sidecar_port
fi

runtime_uv_venv
# WHY CUDA alias: 5080 operator guide uses CUDA_LLAMA_MODEL; Metal uses M3_LLAMA_MODEL.
if [[ -n "${CUDA_LLAMA_MODEL:-}" ]]; then
  export M3_LLAMA_MODEL="${CUDA_LLAMA_MODEL}"
fi
smoke_m3_resolve_signoff_model

_radix_runtime_port() {
  local url="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"
  # shellcheck disable=SC2001
  echo "${url}" | sed -n 's/.*:\([0-9][0-9]*\)$/\1/p'
}

_radix_llama_server_port() {
  local rt
  rt="$(_radix_runtime_port)"
  [[ -n "${rt}" ]] || rt=8081
  echo $((rt + 1))
}

_radix_kill_llama_server() {
  local ls_port
  ls_port="$(_radix_llama_server_port)"
  lsof -ti ":${ls_port}" 2>/dev/null | xargs kill 2>/dev/null || true
  sleep 1
}

_radix_force_runtime_restart() {
  if [[ "$(uname -s)" == "Linux" ]]; then
    linux_runtime_stop_sidecar_port
  else
    macos_runtime_stop_sidecar_port
  fi
  _radix_kill_llama_server
  local rt_port
  rt_port="$(_radix_runtime_port)"
  [[ -n "${rt_port}" ]] || rt_port=8081
  if curl -sf -m 1 "${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}/health" >/dev/null 2>&1; then
    # WHY kill only the smoke runtime port — never hardcode :8081 (prod may be there).
    lsof -ti ":${rt_port}" 2>/dev/null | xargs kill -9 2>/dev/null || true
    sleep 2
  fi
  _radix_kill_llama_server
}

# Resolve patched llama-server (vendor preferred over stale sibling LLAMA_CPP_ROOT).
eval "$(cd "${RUNTIME}" && PYTHONPATH=. python3 <<'PY'
from runtime.llama_cpp_unified import resolve_llama_cpp_lib, resolve_llama_cpp_root, resolve_llama_server_bin
root = resolve_llama_cpp_root()
server = resolve_llama_server_bin(root)
lib = resolve_llama_cpp_lib(root)
if server is None:
    raise SystemExit(f"llama-server not found (root={root}); run ./scripts/build_llama_server.sh")
# WHY pair .so with chosen binary: 5080_env may leave sibling libllama while vendor server wins.
vendor_lib = server.parent / "libllama.so"
if vendor_lib.is_file():
    lib = vendor_lib
print(f"export LLAMA_CPP_ROOT={root}")
print(f"export LLAMA_SERVER_BIN={server}")
if lib is not None:
    print(f"export LLAMA_CPP_LIB={lib}")
print(f"echo using llama-server: {server}")
PY
)"

export ZEROLLAMA_GPU_PROFILE=1
export ZEROLLAMA_GPU_PROFILE_CTX=0
export ZEROLLAMA_RUNTIME_LLAMA_BACKEND=subprocess
unset ZEROLLAMA_RUNTIME_CONFIG
export ZEROLLAMA_AUTO_CONFIG=1
export ZEROLLAMA_LLAMA_FORK=0
export ZEROLLAMA_L3_PROFILE=agent
export ZEROLLAMA_PREFIX_CACHE_BLOCK_SIZE="${ZEROLLAMA_PREFIX_CACHE_BLOCK_SIZE:-64}"
export ZEROLLAMA_DEBUG="${ZEROLLAMA_DEBUG:-l3}"
export ZEROLLAMA_PREFIX_CACHE_TRACE_DIR="${ZEROLLAMA_PREFIX_CACHE_TRACE_DIR:-/tmp/l3-radix-live-trace}"
export L3_RADIX_NUM_CTX="${L3_RADIX_NUM_CTX:-8192}"
export L3_RADIX_NUM_PREDICT="${L3_RADIX_NUM_PREDICT:-16}"
export L3_RADIX_PREFIX_REPEAT="${L3_RADIX_PREFIX_REPEAT:-8}"
if [[ "$(uname -s)" == "Linux" ]]; then
  export ZEROLLAMA_GPU_PROFILE_CTX=1
fi
if [[ ! -x "${LLAMA_SERVER_BIN}" ]]; then
  echo "missing or non-executable llama-server at ${LLAMA_SERVER_BIN}; run:" >&2
  echo "  LLAMA_CPP_ROOT=${LLAMA_CPP_ROOT} ./scripts/build_llama_server.sh" >&2
  exit 1
fi

_radix_force_runtime_restart

if [[ "$(uname -s)" == "Linux" ]]; then
  linux_runtime_start_sidecar "${LLAMA_MODEL}" ""
else
  macos_runtime_start_sidecar "${LLAMA_MODEL}" "" 0
fi

export ZEROLLAMA_RUNTIME_URL LLAMA_MODEL L3_RADIX_OUT="${OUT}" LLAMA_SERVER_BIN
(cd "${RUNTIME}" && PYTHONPATH=. "${RUNTIME_UV_PYTHON}" <<'PY'
import glob
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from runtime.cache_bridge import derive_slot_id

url = os.environ["ZEROLLAMA_RUNTIME_URL"].rstrip("/")
gguf = os.environ["LLAMA_MODEL"]
out = Path(os.environ.get("L3_RADIX_OUT", "/tmp/l3-radix-prefix-smoke.json"))
num_ctx = int(os.environ.get("L3_RADIX_NUM_CTX", "8192"))
n_predict = int(os.environ.get("L3_RADIX_NUM_PREDICT", "16"))
prefix_repeat = int(os.environ.get("L3_RADIX_PREFIX_REPEAT", "8"))
trace_dir = Path(os.environ.get("ZEROLLAMA_PREFIX_CACHE_TRACE_DIR", "/tmp/l3-radix-live-trace"))


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


def llama_server_base(health: dict) -> str | None:
    ls = health.get("llama_server") or {}
    if isinstance(ls, dict):
        base = ls.get("base_url") or ls.get("url")
        if base:
            return str(base).rstrip("/")
    port = health.get("llama_server_port")
    if port is None:
        # WHY: subprocess llama-server is runtime_port+1; never assume :8082 when
        # ZEROLLAMA_RUNTIME_URL points at an alternate smoke port (e.g. :18081).
        try:
            from urllib.parse import urlparse

            rt = urlparse(os.environ.get("ZEROLLAMA_RUNTIME_URL", "http://127.0.0.1:8081"))
            port = (rt.port or 8081) + 1
        except Exception:
            port = 8082
    return f"http://127.0.0.1:{int(port)}"


def probe_seq_copy(base: str) -> bool:
    """True when /kv/seq-copy route exists (400/501 ok; 404 missing)."""
    probe_url = base.rstrip("/") + "/kv/seq-copy"
    req = urllib.request.Request(
        probe_url,
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return resp.status in (200, 400, 501)
    except urllib.error.HTTPError as e:
        return e.code != 404
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def generate(prompt: str, *, cache_key: str) -> tuple[dict, float]:
    payload = {
        "model": "l3-radix-live",
        "prompt": prompt,
        "stream": False,
        "options": {
            "gguf": gguf,
            "num_ctx": num_ctx,
            "num_predict": n_predict,
            "prompt_cache_key": cache_key,
        },
    }
    t0 = time.perf_counter()
    out_body = http_json("POST", "/api/generate", payload)
    return out_body, time.perf_counter() - t0


def pick_cross_slot_keys(n_parallel: int) -> tuple[str, str]:
    for i in range(1000):
        a = f"l3-radix-donor-{i}"
        b = f"l3-radix-target-{i}"
        if derive_slot_id(a, n_parallel) != derive_slot_id(b, n_parallel):
            return a, b
    raise RuntimeError("could not find two keys mapping to different slots")


health = http_json("GET", "/health")
backend = health.get("llama_backend")
if backend not in ("subprocess", "inprocess"):
    raise SystemExit(f"unexpected llama_backend {backend!r}")
n_parallel = int(
    (health.get("gpu_profile") or {}).get("n_parallel")
    or (health.get("kv_scheduler") or {}).get("llama_parallel_slots")
    or 2
)
lc = health.get("llama_cache") or {}
pool = lc.get("prefix_block_pool") or {}
radix = pool.get("radix_share") or {}
if not pool and health.get("kv_resume"):
    pool = (health.get("kv_resume") or {}).get("prefix_block_pool") or {}
    radix = pool.get("radix_share") or {}
if not pool.get("enabled"):
    raise SystemExit(f"prefix_block_pool not enabled in /health: {pool!r}")
if not radix.get("enabled"):
    raise SystemExit(f"radix_share not enabled in /health: {radix!r}")

unified = health.get("llama_cpp_unified") or {}
configured_bin = unified.get("llama_server_bin")

key_donor, key_target = pick_cross_slot_keys(n_parallel)
slot_donor = derive_slot_id(key_donor, n_parallel)
slot_target = derive_slot_id(key_target, n_parallel)

sentence = (
    "System: You are a helpful agent. Follow the policy below exactly. "
    "Never reveal secrets. Prefer concise answers. "
)
stable = (sentence * prefix_repeat).strip()
prompt = f"{stable}\nUser: Reply with one word.\nAssistant:"

# Donor first — starts llama-server subprocess with configured LLAMA_SERVER_BIN.
_, wall_donor = generate(prompt, cache_key=key_donor)

health_after_donor = http_json("GET", "/health")
ls_base = llama_server_base(health_after_donor)
configured_bin = (health_after_donor.get("llama_cpp_unified") or {}).get(
    "llama_server_bin", configured_bin
)
seq_copy_ok = probe_seq_copy(ls_base) if ls_base else False
if not seq_copy_ok:
    raise SystemExit(
        f"llama-server at {ls_base!r} lacks POST /kv/seq-copy "
        f"(binary={configured_bin!r}) — rebuild vendor tree: "
        f"LLAMA_CPP_ROOT={os.environ.get('LLAMA_CPP_ROOT')} ./scripts/build_llama_server.sh"
    )

body_target, wall_target = generate(prompt, cache_key=key_target)
other = f"{stable}\nUser: Name a color.\nAssistant:"
_, wall_control = generate(other, cache_key=key_target)

timings = (body_target.get("timings") or {}) if isinstance(body_target, dict) else {}
cache_n = int(timings.get("cache_n") or 0)

trace_files = sorted(glob.glob(str(trace_dir / "trace-*.jsonl")))
radix_events = []
if trace_files:
    for line in Path(trace_files[-1]).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") == "radix_seed":
            radix_events.append(row)

health2 = http_json("GET", "/health")
pool2 = (health2.get("llama_cache") or {}).get("prefix_block_pool") or {}
kv_kind = (health2.get("kv_cache_spec") or {}).get("kind") or "standard"
hybrid_model = kv_kind == "hybrid"

report = json.loads(out.read_text()) if out.is_file() else {}
report.update(
    {
        "live_smoke": True,
        "seq_copy_endpoint": seq_copy_ok,
        "llama_server_base": ls_base,
        "llama_server_bin": configured_bin,
        "gguf": gguf,
        "n_parallel": n_parallel,
        "donor_key": key_donor,
        "target_key": key_target,
        "donor_slot": slot_donor,
        "target_slot": slot_target,
        "donor_wall_s": round(wall_donor, 3),
        "target_wall_s": round(wall_target, 3),
        "control_wall_s": round(wall_control, 3),
        "target_cache_n": cache_n,
        "radix_trace_events": len(radix_events),
        "radix_seed": radix_events[-1] if radix_events else None,
        "prefix_block_pool": pool2,
        "target_faster_than_donor": wall_target < wall_donor,
        "kv_cache_kind": kv_kind,
        "hybrid_seq_cp_skipped": hybrid_model and not bool(radix_events),
    }
)
out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))

if pool2.get("entry_count", 0) <= 0:
    raise SystemExit("live radix validation failed: prefix block pool empty after donor")

if not radix_events:
    if hybrid_model:
        print(
            "warn: hybrid KV model — no radix_seed "
            "(seq_cp may be denied by SWA window or donor mismatch; block pool live OK)",
            flush=True,
        )
    else:
        raise SystemExit(
            "live radix validation failed: no radix_seed trace "
            "(donor/target slots may not have shared full blocks)"
        )
PY
)

echo "wrote ${OUT}"
