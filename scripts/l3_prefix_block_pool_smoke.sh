#!/usr/bin/env bash
# L3 prefix block pool + optional LMCache tier — offline policy gate (no GPU).
#
# WHY: hash-chained prefix verification and LMCache metadata tier are env-gated;
# this script runs pytest + a minimal replay without llama-server.
#
# Usage:
#   ./scripts/l3_prefix_block_pool_smoke.sh
# Optional live (needs runtime + model):
#   M3_LLAMA_MODEL=/path/model.gguf L3_BLOCK_POOL_LIVE=1 ./scripts/l3_prefix_block_pool_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}/runtime"

OUT="${L3_BLOCK_POOL_OUT:-/tmp/l3-prefix-block-pool-smoke.json}"

echo "== prefix block pool pytest (no GPU) =="
PYTHONPATH=. python3 -m pytest \
  tests/test_prefix_block_pool.py \
  tests/test_prefix_cache_policy.py \
  tests/test_cache_bridge.py \
  -q

echo "== prefix block pool policy replay =="
PYTHONPATH=. python3 <<PY
import json
import os
from pathlib import Path

from runtime.kv.prefix_block_pool import (
    build_model_scope,
    get_prefix_block_pool,
    reset_prefix_block_pools_for_tests,
)
from runtime.kv.lmcache_tier import reset_lmcache_tier_for_tests
from runtime.prefix_cache_policy import prefix_cache_decision, resolve_prefix_cache_policy

os.environ["ZEROLLAMA_PREFIX_BLOCK_POOL"] = "1"
reset_prefix_block_pools_for_tests()
reset_lmcache_tier_for_tests()

policy = resolve_prefix_cache_policy(spec_method="none")
scope = build_model_scope(model_hash="smoke-model")
pool = get_prefix_block_pool(model_scope=scope)
tokens = list(range(100, 100 + 1024))
pool.register_prefix(
    tokens,
    scope=scope,
    seq_pos=1024,
    session_key="smoke-key",
    slot_id=0,
)

allow, resume, reason = prefix_cache_decision(
    "smoke-key",
    policy,
    seq_pos=1024,
    prompt_tokens=1024,
    prompt_token_ids=tokens,
    model_hash="smoke-model",
)
assert allow is True and resume == 1024 and reason is None

drift = list(tokens)
drift[700] = 999
allow2, _, reason2 = prefix_cache_decision(
    "smoke-key",
    policy,
    seq_pos=1024,
    prompt_tokens=1024,
    prompt_token_ids=drift,
    model_hash="smoke-model",
)
assert allow2 is False and reason2 == "prefix_block_hash_mismatch"

report = {
    "offline_replay": "ok",
    "matched_blocks": pool.lookup_longest_prefix(
        tokens, scope=scope, seq_pos=1024
    ).matched_blocks,
    "drift_denied": reason2,
}
Path("${OUT}").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY

if [[ "${L3_BLOCK_POOL_LIVE:-0}" != "1" ]]; then
  echo "wrote ${OUT} (set L3_BLOCK_POOL_LIVE=1 for two-turn runtime smoke)"
  exit 0
fi

# shellcheck source=scripts/runtime_uv_venv.sh
source "${ROOT}/scripts/runtime_uv_venv.sh"
# shellcheck source=scripts/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime_smoke_lib.sh"
if [[ "$(uname -s)" == "Linux" ]]; then
  # shellcheck source=scripts/linux_runtime_serve_lib.sh
  source "${ROOT}/scripts/linux_runtime_serve_lib.sh"
  linux_runtime_urls
  trap linux_runtime_sidecar_cleanup EXIT
  linux_runtime_start_sidecar "" ""
else
  # shellcheck source=scripts/macos_runtime_serve_lib.sh
  source "${ROOT}/scripts/macos_runtime_serve_lib.sh"
  macos_runtime_urls
  trap macos_runtime_sidecar_cleanup EXIT
  macos_runtime_start_sidecar "" "" 0
fi

runtime_uv_venv
smoke_m3_resolve_signoff_model
export ZEROLLAMA_L3_PROFILE="${ZEROLLAMA_L3_PROFILE:-agent}"
unset ZEROLLAMA_RUNTIME_CONFIG
export ZEROLLAMA_LMCACHE_URI="${ZEROLLAMA_LMCACHE_URI:-file://${HOME}/.cache/zerollama/lmcache-smoke}"

(cd "${ROOT}/runtime" && PYTHONPATH=. "${RUNTIME_UV_PYTHON}" <<'PY'
import json
import os
import time
import urllib.request
from pathlib import Path

url = os.environ["ZEROLLAMA_RUNTIME_URL"].rstrip("/")
gguf = os.environ["LLAMA_MODEL"]
cache_key = os.environ.get("L3_CACHE_KEY", "l3-block-pool-smoke")
out = Path(os.environ.get("L3_BLOCK_POOL_OUT", "/tmp/l3-prefix-block-pool-smoke.json"))

def post(path, body):
    req = urllib.request.Request(
        f"{url}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode())

health = json.loads(urllib.request.urlopen(f"{url}/health", timeout=30).read())
pool = (health.get("llama_cache") or {}).get("prefix_block_pool") or {}
if not pool.get("enabled"):
    raise SystemExit("prefix_block_pool not enabled in /health.llama_cache")

prompt = "User: Reply with one word.\nAssistant:"
opts = {"gguf": gguf, "num_ctx": 8192, "num_predict": 8, "prompt_cache_key": cache_key}
t0 = time.perf_counter()
post("/api/generate", {"model": "l3-block-pool", "prompt": prompt, "stream": False, "options": opts})
wall1 = time.perf_counter() - t0
t1 = time.perf_counter()
post("/api/generate", {"model": "l3-block-pool", "prompt": prompt, "stream": False, "options": opts})
wall2 = time.perf_counter() - t1
health2 = json.loads(urllib.request.urlopen(f"{url}/health", timeout=30).read())
pool2 = (health2.get("llama_cache") or {}).get("prefix_block_pool") or {}
report = json.loads(out.read_text()) if out.is_file() else {}
report.update({
    "live_smoke": True,
    "first_wall_s": round(wall1, 3),
    "second_wall_s": round(wall2, 3),
    "prefix_block_pool": pool2,
})
out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
)

echo "wrote ${OUT}"
