#!/usr/bin/env bash
# Phase 15 v0: build native KV extension + parity pytest (no GPU).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}/runtime"

echo "== Phase 15: llama-kv-ext pin check =="
"${ROOT}/scripts/phase15_llama_kv_ext_pin_check.sh"

echo "== Phase 15: llama.cpp patch doctor =="
"${ROOT}/scripts/llama_patch_doctor.sh"

echo "== Phase 15: build native extension (in-tree) =="
# WHY ZEROLLAMA_KV_DECODE_LOOP=0 on CI default: GitHub runners have no libllama;
# auto-link (v25) is skipped. GPU sign-off scripts rebuild linked ext (clean build/).
(
  cd "${ROOT}/runtime"
  ZEROLLAMA_KV_DECODE_LOOP="${ZEROLLAMA_KV_DECODE_LOOP:-0}" python3 setup.py build_ext --inplace >/dev/null
)

echo "== Phase 15: KV pytest (native + bind + physical + tick) =="
PYTHONPATH=.:${PYTHONPATH:-} python3 -m pytest \
  tests/test_kv_native_parity.py \
  tests/test_block_pool.py \
  tests/test_kv_bind.py \
  tests/test_kv_accounting.py \
  tests/test_kv_physical.py \
  tests/test_kv_native_tick.py \
  tests/test_kv_native_decode.py \
  tests/test_kv_forward_plan.py \
  tests/test_kv_decode_plan.py \
  tests/test_kv_decode_work_plan.py \
  tests/test_kv_native_stats.py \
  tests/test_kv_page_bind.py \
  tests/test_kv_tensor_probe.py \
  tests/test_kv_decode_engine_resume.py \
  tests/test_kv_native_decode_batch.py \
  tests/test_kv_decode_long_ctx.py \
  tests/test_kv_native_build.py \
  tests/test_kv_decode_batch_loop.py \
  tests/test_kv_decode_engine_batch.py \
  tests/test_kv_auto_batch.py \
  tests/test_kv_live_physical.py \
  tests/test_internal_kv_snapshot.py \
  tests/test_resolve_parallel_slots.py \
  tests/test_prefix_cache_policy.py \
  tests/test_kv_cache_spec.py \
  tests/test_prefix_cache_trace.py \
  tests/test_prefix_cache_golden_trace.py \
  tests/test_spec_bind.py \
  tests/test_decode_graph_policy.py \
  tests/test_subprocess_slot_state.py \
  tests/test_cache_bridge.py \
  tests/test_prefix_block_pool.py \
  tests/test_radix_prefix_share.py \
  tests/test_radix_seq_copy.py \
  tests/test_radix_engine_guard.py \
  -q

echo "== Phase 15: health KV smoke =="
"${ROOT}/scripts/phase15_health_smoke.sh"

echo "== Phase 15: backend factory (native) =="
PYTHONPATH=.:${PYTHONPATH:-} ZEROLLAMA_RUNTIME_KV_NATIVE=1 python3 -c "
from runtime.kv.backend import kv_backend_health, create_block_pool
h = kv_backend_health()
assert h['backend'] == 'native' and h['native_available']
p = create_block_pool(num_blocks=4, block_size=16, device_id=0)
assert p.allocate(1)
print('kv:', h)
"

echo "PASS: phase15_kv_native_ci"
