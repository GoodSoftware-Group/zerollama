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
  runtime_uv_venv.sh
  gpu_phase13_snapshot.sh
  gpu_clamp_smoke.sh
  phase12_capture_tool_transcript.sh
  gpu_5080_session.sh
  gpu_harmony_capture.sh
  macos_metal_smoke.sh
  gpu_metal_session.sh
  m3_metal_signoff.sh
  metal_signoff.sh
  mac_setup.sh
  mac_cgo_env.sh
  build_zerollama_mac.sh
  build_mlx_dylibs_mac.sh
  build_production_mac.sh
  ensure_mlx_sources.sh
  training_uv_venv.sh
  serve_mac_runtime.sh
  phase15_metal_signoff.sh
  macos_runtime_serve_lib.sh
  l2_fork_eval.sh
  l2_metal_bench.sh
  build_eliza_llama_server.sh
  phase14_backend_smoke.sh
  phase14_inprocess_smoke.sh
  phase14_wheel_cpu_smoke.sh
  phase14_yaml_config_smoke.sh
  phase14_yaml_config_full_smoke.sh
  phase14_5080_signoff.sh
  phase14_subprocess_default_smoke.sh
  phase14_wheel_gpu_smoke.sh
  phase14_enable_yaml_inprocess.sh
  phase14_both_backends.sh
  phase14_serve_env.sh
  phase15_inprocess_kv_smoke.sh
  phase15_inprocess_multiseq_smoke.sh
  phase15_inprocess_signoff.sh
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
grep -q 'metal-unified' "${ROOT}/runtime/runtime/gpu_vram.py"
grep -q 'apple_silicon.yaml' "${ROOT}/runtime/runtime/autoconfig.py"
grep -q 'read_host_memory()' "${ROOT}/runtime/runtime/host_memory.py"
grep -q 'test_host_memory_darwin.py' "${ROOT}/scripts/macos_metal_smoke.sh"
grep -q 'runtime_uv_venv.sh' "${ROOT}/scripts/macos_metal_smoke.sh"
grep -q 'macos_runtime_serve_lib.sh' "${ROOT}/scripts/m3_metal_signoff.sh"
grep -q 'phase14_yaml_config_smoke.sh' "${ROOT}/scripts/m3_metal_signoff.sh"
grep -q 'macos_runtime_serve_lib.sh' "${ROOT}/scripts/serve_mac_runtime.sh"
grep -q 'macos_runtime_start_sidecar' "${ROOT}/scripts/serve_mac_runtime.sh"
grep -q 'phase15_metal_signoff.sh' "${ROOT}/scripts/m3_metal_signoff.sh"
grep -q 'smoke_m3_resolve_signoff_model' "${ROOT}/scripts/runtime_smoke_lib.sh"
grep -q 'smoke_m3_resolve_signoff_model' "${ROOT}/scripts/m3_metal_signoff.sh"
grep -q 'PHASE15_SKIP_BOOT' "${ROOT}/scripts/phase15_metal_signoff.sh"
grep -q 'macos_runtime_serve_lib.sh' "${ROOT}/scripts/serve_mac_runtime.sh"
grep -q 'RUN_E2E_PHASE15' "${ROOT}/scripts/gpu_metal_session.sh"
grep -q 'METAL_SELF_START' "${ROOT}/scripts/gpu_metal_session.sh"
grep -q 'RUN_E2E_PHASE15=1' "${ROOT}/scripts/metal_signoff.sh"
grep -q 'RUN_E2E_QWEN35' "${ROOT}/scripts/qwen35_mac_smoke.sh"
grep -q 'qwen35_mac_smoke.sh' "${ROOT}/scripts/m3_metal_signoff.sh"
grep -q 'test_m3_model_picker.py' "${ROOT}/scripts/macos_metal_smoke.sh"
grep -q 'NewDoctorCommand' "${ROOT}/cmd/cmd.go"
grep -q 'mac_setup.sh' "${ROOT}/docs/development.md"
grep -q 'dev_bootstrap.sh' "${ROOT}/scripts/dev_bootstrap.sh"
grep -q 'ensure_llama_cpp_sibling' "${ROOT}/scripts/mac_setup.sh"
grep -q 'MAC_SETUP_SIGNOFF:-0' "${ROOT}/scripts/mac_setup.sh"
grep -q 'M14' "${ROOT}/docs/ROADMAP.md"
grep -q 'Onboarding tiers' "${ROOT}/docs/apple-silicon-metal.md"
grep -q 'llama_backend_fallback' "${ROOT}/runtime/runtime/engine.py"
grep -q 'training_uv_venv.sh' "${ROOT}/scripts/mac_setup.sh"
grep -q 'doctor --fix' "${ROOT}/cmd/doctor.go"
grep -q 'macos-darwin-smoke' "${ROOT}/.github/workflows/zerollama-regression.yaml"
grep -q 'training/.venv-training' "${ROOT}/cmd/doctor.go"
grep -q 'checkTrainingQloraPayload' "${ROOT}/server/training_platform.go"
grep -q 'zerollama serve' "${ROOT}/cmd/doctor.go"
grep -q 'BootstrapDarwinSidecar' "${ROOT}/server/routes.go"
grep -q 'mac_cgo_env' "${ROOT}/scripts/build_zerollama_mac.sh"
grep -q 'mac-dev-setup.md' "${ROOT}/docs/development.md"
grep -q 'build_zerollama_mac' "${ROOT}/cmd/doctor.go"
grep -q 'build_production_mac' "${ROOT}/docs/mac-dev-setup.md"
grep -q 'BUILD_MLX' "${ROOT}/scripts/build_zerollama_mac.sh"
grep -q 'pickOllamaEngine' "${ROOT}/llm/server.go"
grep -q 'Persistent()' "${ROOT}/kvcache/causal.go"
grep -q 'darwinSidecarEnabled' "${ROOT}/server/darwin_sidecar.go"
grep -q 'zerollama serve' "${ROOT}/docs/development.md"
grep -q 'runtime_url_port' "${ROOT}/scripts/runtime_smoke_lib.sh"
grep -q 'llama_backend: inprocess' "${ROOT}/runtime/configs/apple_silicon.yaml"
grep -q 'vm.swapusage' "${ROOT}/runtime/runtime/host_memory.py"
grep -q 'apple_silicon' "${ROOT}/runtime/runtime/gpu_snapshot.py"
grep -q 'gpu_metal_session' "${ROOT}/docs/apple-silicon-metal.md"
grep -q 'l2_metal_bench' "${ROOT}/docs/gpu-profiles-l2.md"
grep -q 'ZEROLLAMA_RUNTIME_LLAMA_BACKEND=subprocess' "${ROOT}/scripts/l2_metal_bench.sh"
grep -q 'l2_metal_bench' "${ROOT}/scripts/l2_fork_eval.sh"
grep -q 'IsMLX()' "${ROOT}/docs/mlx-routing-policy.md"
grep -q 'modelUsesRuntimeInference' "${ROOT}/server/runtime_inference_routing.go"
grep -q 'RUN_E2E_LEGACY=1 with RUN_E2E_GPU' "${ROOT}/scripts/e2e_runtime_smoke.sh"
grep -q 'RUN_E2E_VRAM_CLAMP' "${ROOT}/scripts/gpu_clamp_smoke.sh"
grep -q 'phase12_capture_tool_transcript' "${ROOT}/scripts/phase12_capture_tool_transcript.sh" || grep -q 'X-Zerollama-Runtime' "${ROOT}/scripts/phase12_capture_tool_transcript.sh"
grep -q 'RUN_E2E_PHASE14' "${ROOT}/scripts/e2e_runtime_smoke.sh"
grep -q 'smoke_runtime_needs_server_bin' "${ROOT}/scripts/runtime_smoke_lib.sh"
grep -q 'internal/tokenize' "${ROOT}/scripts/e2e_runtime_smoke.sh"
grep -q 'smoke_llama_model_config_hint' "${ROOT}/scripts/runtime_smoke_lib.sh"
grep -q 'truncate_mode=tokenize' "${ROOT}/scripts/e2e_runtime_smoke.sh"
grep -q 'smoke_runtime_require_phase14_endpoints' "${ROOT}/scripts/runtime_smoke_lib.sh"
grep -q 'smoke_runtime_assert_llama_backend_source' "${ROOT}/scripts/runtime_smoke_lib.sh"
grep -q 'skip_global_vram_factor_export' "${ROOT}/runtime/runtime/gpu_health_report.py"
grep -q 'skip_global_vram_factor_export' "${ROOT}/runtime/runtime/gpu_snapshot.py"
grep -q 'vram_recommendations' "${ROOT}/runtime/runtime/gpu_health_report.py"
grep -q 'skip_global_vram_factor_export' "${ROOT}/runtime/tests/test_vram_recommendations.py"
grep -q 'smoke_runtime_apply_backend_flags_from_health' "${ROOT}/scripts/runtime_smoke_lib.sh"
grep -q 'inferred from /health' "${ROOT}/scripts/phase14_yaml_config_smoke.sh"
grep -q 'RUN_E2E_LLAMA_BACKEND_SOURCE=env' "${ROOT}/scripts/phase14_inprocess_smoke.sh"
grep -q 'RUN_E2E_INPROCESS=1' "${ROOT}/scripts/phase14_inprocess_smoke.sh"
grep -q 'RUN_E2E_LLAMA_CPP_PYTHON=1' "${ROOT}/scripts/phase14_wheel_cpu_smoke.sh"
grep -q 'RUN_E2E_LLAMA_BACKEND_SOURCE=env' "${ROOT}/scripts/phase14_wheel_cpu_smoke.sh"
grep -q 'RUN_E2E_LLAMA_BACKEND_SOURCE=config' "${ROOT}/scripts/phase14_yaml_config_smoke.sh"
grep -q 'RUN_E2E_LLAMA_BACKEND_SOURCE=default' "${ROOT}/scripts/phase14_subprocess_default_smoke.sh"
grep -q 'canonical_llama_backend' "${ROOT}/runtime/runtime/worker/factory.py"
grep -q 'llama_backend_from_file' "${ROOT}/runtime/runtime/config.py"
grep -q 'RUN_E2E_PHASE14_SIGNOFF' "${ROOT}/scripts/gpu_5080_session.sh"
grep -q 'RUN_E2E_PHASE15' "${ROOT}/scripts/gpu_5080_session.sh"
grep -q '_saved_phase14_signoff' "${ROOT}/scripts/gpu_5080_session.sh"
grep -q 'phase14_5080_signoff.sh' "${ROOT}/scripts/gpu_smoke_all.sh"
grep -q 'phase15_inprocess_signoff.sh' "${ROOT}/scripts/gpu_smoke_all.sh"
grep -q 'phase15_inprocess_kv_smoke.sh' "${ROOT}/scripts/phase15_inprocess_signoff.sh"
grep -q 'phase15_inprocess_multiseq_smoke.sh' "${ROOT}/scripts/phase15_inprocess_signoff.sh"
grep -q 'RUN_E2E_PHASE14' "${ROOT}/scripts/gpu_smoke_all.sh"
grep -q 'llama_cpp_wheel_health' "${ROOT}/runtime/runtime/worker/llama_cpp_python.py"
grep -q 'llama_cpp' "${ROOT}/scripts/gpu_phase13_snapshot.sh"
grep -q 'smoke_runtime_assert_llama_cpp_gpu' "${ROOT}/scripts/runtime_smoke_lib.sh"
grep -q 'RUN_E2E_LLAMA_CPP_PYTHON_GPU=1' "${ROOT}/scripts/phase14_wheel_gpu_smoke.sh"
grep -q 'phase14_inprocess_smoke.sh' "${ROOT}/scripts/phase14_both_backends.sh"
grep -q 'phase14_wheel_cpu_smoke.sh' "${ROOT}/scripts/phase14_both_backends.sh"
grep -q 'checkLoopbackPortFree' "${ROOT}/x/runtimeworker/client.go"
grep -q 'embed_boot' "${ROOT}/runtime/runtime/engine.py"
grep -q 'kv_decode_steps (generate)' "${ROOT}/scripts/e2e_runtime_smoke.sh"
grep -q 'phase14_yaml_config_full_smoke.sh' "${ROOT}/scripts/phase14_5080_signoff.sh"
grep -q 'phase15_inprocess_signoff.sh' "${ROOT}/scripts/phase14_5080_signoff.sh"
grep -q 'phase14_both_backends.sh' "${ROOT}/scripts/phase14_5080_signoff.sh"
grep -q 'phase15_inprocess_kv_smoke.sh' "${ROOT}/scripts/check_gpu_scripts.sh"
grep -q 'phase14_backend_smoke.sh' "${ROOT}/scripts/phase15_inprocess_kv_smoke.sh"
grep -q 'smoke_runtime_assert_kv_snapshot' "${ROOT}/scripts/runtime_smoke_lib.sh"
grep -q 'smoke_runtime_assert_kv_snapshot' "${ROOT}/scripts/phase15_inprocess_kv_smoke.sh"
grep -q 'smoke_runtime_assert_kv_snapshot' "${ROOT}/scripts/phase15_inprocess_multiseq_smoke.sh"
grep -q 'llama_parallel_slots: 2' "${ROOT}/scripts/phase15_inprocess_multiseq_smoke.sh"
grep -q 'ZEROLLAMA_RUNTIME_CONFIG' "${ROOT}/scripts/phase14_yaml_config_full_smoke.sh"
grep -q 'phase14_yaml_config_smoke.sh' "${ROOT}/scripts/phase14_yaml_config_full_smoke.sh"
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
