#!/usr/bin/env bash
# Syntax-check GPU operator scripts (CI-friendly, no GPU required).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scripts=(
  e2e_runtime_smoke.sh
  e2e_coordination_smoke.sh
  gpu_smoke_all.sh
  gpu_health_report.sh
  runtime_vram_estimate.sh
  serve_gpu_example.sh
  phase12_golden_ci.sh
  runtime_smoke_lib.sh
  gpu_phase13_snapshot.sh
  gpu_clamp_smoke.sh
  phase12_capture_tool_transcript.sh
  gpu_5080_session.sh
  gpu_harmony_capture.sh
  phase14_backend_smoke.sh
  phase14_both_backends.sh
  phase14_serve_env.sh
  e2e_training_ops_smoke.sh
  repro_shared_interpreter_health_hang.sh
  phase15_kv_native_ci.sh
  phase15_health_smoke.sh
)

for s in "${scripts[@]}"; do
  bash -n "${ROOT}/scripts/${s}"
  echo "ok: scripts/${s}"
done

cd "${ROOT}/runtime"
PYTHONPATH=. python3 -c "from runtime.gpu_health_report import format_gpu_health_tuning_report; print('ok: runtime.gpu_health_report')"

# Smoke script must include proxy tools path when RUN_E2E_TOOLS is documented.
grep -q 'proxy tools chat' "${ROOT}/scripts/e2e_runtime_smoke.sh"
grep -q 'proxy v1 tools chat' "${ROOT}/scripts/e2e_runtime_smoke.sh"
grep -q 'v1 tools chat:' "${ROOT}/scripts/e2e_runtime_smoke.sh"
grep -q 'RUN_E2E_LEGACY' "${ROOT}/scripts/e2e_runtime_smoke.sh"
grep -q 'phase12_golden_ci' "${ROOT}/scripts/gpu_smoke_all.sh"
grep -q 'runtime_resume_if_needed' "${ROOT}/scripts/gpu_smoke_all.sh"
grep -q 'runtime_smoke_lib.sh' "${ROOT}/scripts/e2e_runtime_smoke.sh"
grep -q 'smoke_prepare_vram_for_runtime' "${ROOT}/scripts/gpu_smoke_all.sh"
grep -q 'smoke_unload_ggml_runners' "${ROOT}/scripts/runtime_smoke_lib.sh"
grep -q 'smoke_ggml_runner_running' "${ROOT}/scripts/runtime_smoke_lib.sh"
grep -q 'recommend_from_snapshot' "${ROOT}/runtime/runtime/gpu_snapshot.py"
grep -q 'apply_vram_defaults_from_config' "${ROOT}/runtime/runtime/vram_yaml_defaults.py"
grep -q 'runtime.gpu_snapshot' "${ROOT}/scripts/gpu_5080_session.sh"
grep -q 'RUN_E2E_LEGACY=1 with RUN_E2E_GPU' "${ROOT}/scripts/e2e_runtime_smoke.sh"
grep -q 'RUN_E2E_VRAM_CLAMP' "${ROOT}/scripts/gpu_clamp_smoke.sh"
grep -q 'phase12_capture_tool_transcript' "${ROOT}/scripts/phase12_capture_tool_transcript.sh" || grep -q 'X-Zerollama-Runtime' "${ROOT}/scripts/phase12_capture_tool_transcript.sh"
grep -q 'RUN_E2E_PHASE14' "${ROOT}/scripts/e2e_runtime_smoke.sh"
grep -q 'smoke_runtime_needs_server_bin' "${ROOT}/scripts/runtime_smoke_lib.sh"
grep -q 'internal/tokenize' "${ROOT}/scripts/e2e_runtime_smoke.sh"
grep -q 'smoke_llama_model_config_hint' "${ROOT}/scripts/runtime_smoke_lib.sh"
grep -q 'truncate_mode=tokenize' "${ROOT}/scripts/e2e_runtime_smoke.sh"
grep -q 'smoke_runtime_require_phase14_endpoints' "${ROOT}/scripts/runtime_smoke_lib.sh"
grep -q '/api/train/status' "${ROOT}/scripts/e2e_training_ops_smoke.sh"
grep -q 'cmd.*ping' "${ROOT}/scripts/e2e_training_ops_smoke.sh"
grep -q 'ZEROLLAMA_RUNTIME_SHARED_PYTHON' "${ROOT}/runtime/runtime/gpu_vram.py"
grep -q 'health try' "${ROOT}/scripts/repro_shared_interpreter_health_hang.sh"
grep -q 'test_kv_native_parity' "${ROOT}/scripts/phase15_kv_native_ci.sh"
grep -q 'kv_backend_health' "${ROOT}/runtime/runtime/engine.py"
grep -q 'native_requested' "${ROOT}/runtime/runtime/kv/backend.py"
grep -q 'kv_scheduler_snapshot' "${ROOT}/runtime/runtime/kv/accounting.py"
grep -q 'kv_bind_health' "${ROOT}/runtime/runtime/kv/bind.py"
grep -q 'assert_kv_capacity' "${ROOT}/runtime/runtime/engine.py"
grep -q 'scheduler_tick' "${ROOT}/runtime/native/kv_block_pool.c"
grep -q 'kv_physical_health' "${ROOT}/runtime/runtime/kv/physical.py"
grep -q 'kv_scheduler_tick' "${ROOT}/runtime/runtime/engine.py"
grep -q 'recent_alignments' "${ROOT}/runtime/runtime/kv/physical.py"
grep -q 'decode_step' "${ROOT}/runtime/native/kv_block_pool.c"
grep -q 'kv_stats' "${ROOT}/runtime/native/kv_block_pool.c"
grep -q 'kv_forward_plan' "${ROOT}/runtime/runtime/kv/forward_plan.py"
grep -q 'kv_snapshot' "${ROOT}/runtime/runtime/engine.py"
grep -q '/internal/kv-snapshot' "${ROOT}/runtime/runtime/server/app.py"
grep -q 'phase15_health_smoke' "${ROOT}/scripts/phase15_kv_native_ci.sh"
grep -q 'record_decode_step' "${ROOT}/runtime/runtime/worker/libllama_ctypes.py"
grep -q 'kv_slot' "${ROOT}/runtime/runtime/scheduler/scheduler.py"
grep -q 'resolve_parallel_slots' "${ROOT}/runtime/runtime/llama_args.py"
grep -q '_effective_llama_parallel_slots' "${ROOT}/runtime/runtime/engine.py"
echo "ok: e2e_runtime_smoke tools markers"

echo "PASS: check_gpu_scripts"
