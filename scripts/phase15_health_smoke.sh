#!/usr/bin/env bash
# Phase 15 v7: /health KV export keys (no GPU, no llama-server).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}/runtime"

PYTHONPATH=. python3 <<'PY'
from runtime.config import RuntimeConfig
from runtime.engine import InferenceEngine
from pathlib import Path

root = Path(".").resolve()
cfg = RuntimeConfig(
    host="127.0.0.1",
    port=8081,
    llama_cpp_root=root,
    llama_server_bin=None,
    llama_model=None,
    num_blocks=64,
    block_size=16,
    device_count=1,
)
eng = InferenceEngine(cfg)
h = eng.health()
for key in (
    "kv",
    "kv_scheduler",
    "kv_bind",
    "kv_forward_plans",
    "kv_page_bind",
    "kv_decode_loop",
    "kv_resume",
    "kv_continuous_batch",
    "kv_auto_batch",
    "kv_live_physical",
    "kv_native_stats",
    "kv_decode_steps",
    "kv_scheduler_tick",
):
    assert key in h, f"missing /health key: {key}"
pb = h["kv_page_bind"]
assert "status" in pb
assert "writable_bind_available" in pb
assert "writable_bind_api" in pb
assert "external_alias_available" in pb
assert "external_alias_api" in pb
ab = h["kv_auto_batch"]
assert "non_stream" in ab and "stream" in ab
for sub in ("non_stream", "stream"):
    for field in ("enabled", "pending", "window_ms", "flush_count", "batched_requests"):
        assert field in ab[sub], (sub, field)
if pb.get("available"):
    assert "bind_level" in pb
    assert "slots" in pb
assert isinstance(h["kv_forward_plans"], list)
snap = eng.kv_snapshot()
assert "kv_forward_plans" in snap
assert "kv_page_migration" in snap
mig = snap.get("kv_page_migration") or {}
if isinstance(mig.get("migration_summary"), dict):
    summary = mig["migration_summary"]
    for key in ("pages_live", "n_layers", "full_plan_endpoint"):
        assert key in summary, key
    assert summary["full_plan_endpoint"] == "/internal/kv-snapshot"
plan = mig.get("plan")
if isinstance(plan, dict) and plan.get("pages"):
    for page in plan["pages"]:
        for layer in page.get("layers") or []:
            for copy_key in ("k_copy", "v_copy"):
                block = layer.get(copy_key)
                if isinstance(block, dict):
                    assert "src_ptr" not in block, copy_key
print("phase15_health_smoke: ok", "kv_backend=", h["kv"].get("backend"))
PY

echo "PASS: phase15_health_smoke"
