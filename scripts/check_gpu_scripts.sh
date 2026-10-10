#!/usr/bin/env bash
# Syntax-check GPU operator scripts (CI-friendly, no GPU required).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Post-reorg: scripts live under scripts/{e2e,gpu,phase,...}/ — resolve by basename.
declare -A _GPU_SCRIPT_PATH=()
while IFS= read -r -d '' f; do
  _GPU_SCRIPT_PATH["$(basename "$f")"]="$f"
done < <(find "${ROOT}/scripts" -type f \( -name '*.sh' -o -name '*.py' \) -print0 2>/dev/null)

spath() {
  local name="$1"
  local p="${_GPU_SCRIPT_PATH[$name]:-}"
  if [[ -z "$p" ]]; then
    echo "missing script under scripts/: $name" >&2
    exit 1
  fi
  printf '%s\n' "$p"
}

scripts=(
  e2e_runtime_smoke.sh
  e2e_coordination_smoke.sh
  gpu_smoke_all.sh
  gpu_health_report.sh
  runtime_vram_estimate.sh
  serve_gpu_example.sh
  serve_production_wrapper.sh
  phase12_golden_ci.sh
  runtime_smoke_lib.sh
  runtime_uv_venv.sh
  gpu_phase13_snapshot.sh
  gpu_clamp_smoke.sh
  phase12_capture_tool_transcript.sh
  gpu_5080_session.sh
  5080_env.sh
  5080_resignoff.sh
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
  l2_cuda_bench.sh
  l1_cuda_calibrate.sh
  l1_cuda_concurrent_bench.sh
  l1_cuda_full_gate.sh
  l1_gate_report.sh
  l1_metal_gate.sh
  l1_full_gate.sh
  l2_gate_report.sh
  l2_runtime_compat_smoke.sh
  l2_cuda_runtime_compat_smoke.sh
  l2_full_gate.sh
  l2_cuda_full_gate.sh
  linux_runtime_serve_lib.sh
  l3_cache_smoke.sh
  l3_spec_cache_smoke.sh
  l3_prefix_cache_trace_replay.sh
  l3_prefix_block_pool_smoke.sh
  l3_radix_prefix_smoke.sh
  l3_gate_report.sh
  l3_production_gate.sh
  l3_cuda_full_gate.sh
  l3_full_gate.sh
  l3_inprocess_smoke.sh
  l3_agent_bench.sh
  build_eliza_llama_server.sh
  phase14_backend_smoke.sh
  phase17_llama_server_smoke.sh
  flash_moe_smoke.sh
  ane_probe_build.sh
  ane_probe_smoke.sh
  build_flash_moe_llama_server.sh
  flash_moe_extract_sidecar.sh
  phase16_edge_smoke.sh
  phase16_edge_build_smoke.sh
  phase16_edge_binary_smoke.sh
  phase11_metal_admission_smoke.sh
  phase13_metal_vram_smoke.sh
  phase11_13_15_metal_signoff.sh
  phase17_linux_auto_smoke.sh
  phase17_l2_pin_status.sh
  phase15_upstream_kv_watch.sh
  build_zerollama_edge.sh
  serve_edge.sh
  serve_linux_auto.sh
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
  phase15_llama_kv_ext_pin_check.sh
  llama_patch_doctor.sh
  phase15_health_smoke.sh
  phase15_migration_summary_smoke.sh
  phase15_stream_auto_batch_smoke.sh
  phase15_auto_batch_smoke.sh
  phase15_auto_batch_signoff.sh
)

for s in "${scripts[@]}"; do
  bash -n "$(spath "$s")"
  echo "ok: $(spath "$s" | sed "s|^${ROOT}/||")"
done

cd "${ROOT}/runtime"
PYTHONPATH=. python3 -c "from runtime.gpu_health_report import format_gpu_health_tuning_report; print('ok: runtime.gpu_health_report')"

# Smoke script must include proxy tools path when RUN_E2E_TOOLS is documented.
grep -q 'proxy tools chat' "$(spath e2e_runtime_smoke.sh)"
grep -q 'proxy v1 tools chat' "$(spath e2e_runtime_smoke.sh)"
grep -q 'v1 tools chat:' "$(spath e2e_runtime_smoke.sh)"
grep -q 'RUN_E2E_LEGACY' "$(spath e2e_runtime_smoke.sh)"
grep -q 'phase12_golden_ci' "$(spath gpu_smoke_all.sh)"
grep -q 'runtime_resume_if_needed' "$(spath gpu_smoke_all.sh)"
grep -q 'runtime_smoke_lib.sh' "$(spath e2e_runtime_smoke.sh)"
grep -q 'smoke_prepare_vram_for_runtime' "$(spath gpu_smoke_all.sh)"
grep -q 'smoke_unload_ggml_runners' "$(spath runtime_smoke_lib.sh)"
grep -q 'smoke_ggml_runner_running' "$(spath runtime_smoke_lib.sh)"
grep -q 'recommend_from_snapshot' "${ROOT}/runtime/runtime/gpu_snapshot.py"
grep -q 'apply_vram_defaults_from_config' "${ROOT}/runtime/runtime/vram_yaml_defaults.py"
grep -q 'runtime.gpu_snapshot' "$(spath gpu_5080_session.sh)"
grep -q 'metal-unified' "${ROOT}/runtime/runtime/gpu_vram.py"
grep -q 'apple_silicon.yaml' "${ROOT}/runtime/runtime/autoconfig.py"
grep -q 'read_host_memory()' "${ROOT}/runtime/runtime/host_memory.py"
grep -q 'test_host_memory_darwin.py' "$(spath macos_metal_smoke.sh)"
grep -q 'runtime_uv_venv.sh' "$(spath macos_metal_smoke.sh)"
grep -q 'macos_runtime_serve_lib.sh' "$(spath m3_metal_signoff.sh)"
grep -q 'phase14_yaml_config_smoke.sh' "$(spath m3_metal_signoff.sh)"
grep -q 'macos_runtime_serve_lib.sh' "$(spath serve_mac_runtime.sh)"
grep -q 'macos_runtime_start_sidecar' "$(spath serve_mac_runtime.sh)"
grep -q 'phase15_metal_signoff.sh' "$(spath m3_metal_signoff.sh)"
grep -q 'smoke_m3_resolve_signoff_model' "$(spath runtime_smoke_lib.sh)"
grep -q 'smoke_m3_resolve_signoff_model' "$(spath m3_metal_signoff.sh)"
grep -q 'PHASE15_SKIP_BOOT' "$(spath phase15_metal_signoff.sh)"
grep -q 'macos_runtime_serve_lib.sh' "$(spath serve_mac_runtime.sh)"
grep -q 'RUN_E2E_PHASE15' "$(spath gpu_metal_session.sh)"
grep -q 'METAL_SELF_START' "$(spath gpu_metal_session.sh)"
grep -q 'RUN_E2E_PHASE15=1' "$(spath metal_signoff.sh)"
grep -q 'RUN_E2E_QWEN35' "$(spath qwen35_mac_smoke.sh)"
grep -q 'qwen35_mac_smoke.sh' "$(spath m3_metal_signoff.sh)"
grep -q 'test_m3_model_picker.py' "$(spath macos_metal_smoke.sh)"
grep -q 'NewDoctorCommand' "${ROOT}/cmd/cmd.go"
grep -q 'mac_setup.sh' "${ROOT}/docs/development.md"
grep -q 'dev_bootstrap.sh' "$(spath dev_bootstrap.sh)"
grep -q 'ensure_llama_cpp_sibling' "$(spath mac_setup.sh)"
grep -q 'MAC_SETUP_SIGNOFF:-0' "$(spath mac_setup.sh)"
grep -q 'M14' "${ROOT}/docs/ROADMAP.md"
grep -q 'Onboarding tiers' "${ROOT}/docs/apple-silicon-metal.md"
grep -q 'llama_backend_fallback' "${ROOT}/runtime/runtime/engine.py"
grep -q 'training_uv_venv.sh' "$(spath mac_setup.sh)"
grep -q 'doctor --fix' "${ROOT}/cmd/doctor.go"
grep -q 'macos-darwin-smoke' "${ROOT}/.github/workflows/zerollama-regression.yaml"
grep -q 'training_embed_build_env.sh' "$(spath 5080_env.sh)"
grep -q 'venv-training/' "${ROOT}/.gitignore"
grep -q 'training/.venv-training' "${ROOT}/cmd/doctor.go"
grep -q 'checkTrainingQloraPayload' "${ROOT}/server/training_platform.go"
grep -q 'zerollama serve' "${ROOT}/cmd/doctor.go"
grep -q 'BootstrapDarwinSidecar' "${ROOT}/server/routes.go"
grep -q 'mac_cgo_env' "$(spath build_zerollama_mac.sh)"
grep -q 'mac-dev-setup.md' "${ROOT}/docs/development.md"
grep -q 'build_zerollama_mac' "${ROOT}/cmd/doctor.go"
grep -q 'build_production_mac' "${ROOT}/docs/mac-dev-setup.md"
grep -q 'BUILD_MLX' "$(spath build_zerollama_mac.sh)"
grep -q 'pickOllamaEngine' "${ROOT}/llm/server_shared.go"
grep -q 'Persistent()' "${ROOT}/kvcache/causal.go"
grep -q 'darwinSidecarEnabled' "${ROOT}/server/darwin_sidecar.go"
grep -q 'zerollama serve' "${ROOT}/docs/development.md"
grep -q 'runtime_url_port' "$(spath runtime_smoke_lib.sh)"
grep -q 'llama_backend: inprocess' "${ROOT}/runtime/configs/apple_silicon.yaml"
grep -q 'vm.swapusage' "${ROOT}/runtime/runtime/host_memory.py"
grep -q 'apple_silicon' "${ROOT}/runtime/runtime/gpu_snapshot.py"
grep -q 'gpu_metal_session' "${ROOT}/docs/apple-silicon-metal.md"
grep -q 'l2_metal_bench' "${ROOT}/docs/gpu-profiles-l2.md"
grep -q 'ZEROLLAMA_RUNTIME_LLAMA_BACKEND=subprocess' "$(spath l2_metal_bench.sh)"
grep -q 'l2_metal_bench' "$(spath l2_fork_eval.sh)"
grep -q 'l2_full_gate' "$(spath m3_metal_signoff.sh)"
grep -q 'l1_cuda_calibrate' "${ROOT}/docs/gpu-profiles-l1.md"
grep -q 'l1_cuda_full_gate' "${ROOT}/docs/gpu-profiles-l1.md"
grep -q 'RUN_E2E_L1' "$(spath gpu_5080_session.sh)"
grep -q 'ZEROLLAMA_GPU_PROFILE' "$(spath l1_cuda_calibrate.sh)"
grep -q 'l2_cuda_bench' "${ROOT}/docs/gpu-profiles-l2.md"
grep -q 'ZEROLLAMA_RUNTIME_LLAMA_BACKEND=subprocess' "$(spath l2_cuda_bench.sh)"
grep -q 'linux_runtime_serve_lib' "$(spath l2_cuda_bench.sh)"
grep -q 'linux_runtime_serve_lib' "$(spath l2_cuda_runtime_compat_smoke.sh)"
grep -q 'c84b3020' "${ROOT}/docs/gpu-profiles-l2.md"
grep -q 'llama-cpp-' "${ROOT}/Makefile.sync"
grep -q 'ensure_llama_vendor_patches' "$(spath build_llama_server.sh)"
grep -q 'ensure_llama_vendor_patches' "$(spath build_zerollama_mac.sh)"
grep -q '_llama_server_binary_ok' "$(spath build_zerollama_mac.sh)"
grep -q 'restore_ane_hook_intree' "$(spath build_zerollama_mac.sh)"
grep -q 'stage_llama_ext_b8_for_vendor' "$(spath ensure_llama_vendor_patches.sh)"
grep -q 'stage_llama_kv_ext_for_vendor' "$(spath ensure_llama_vendor_patches.sh)"
grep -q 'BUILD_RUNTIME_KV_EXT' "$(spath build_zerollama_mac.sh)"
grep -q 'runtime-kv-native.sha' "$(spath build_zerollama_mac.sh)"
grep -q 'kv_native_build_sha' "${ROOT}/server/darwin_sidecar.go"
grep -q '_acquire_llama_server_build_lock' "$(spath build_llama_server.sh)"
grep -q 'rebase_vendor_unified' "$(spath rebase_vendor_unified.sh)"
grep -q 'checkpoint-every-n-tokens' "$(spath build_llama_server.sh)"
grep -q 'l2_cuda_runtime_compat_smoke' "$(spath l2_cuda_full_gate.sh)"
grep -q 'l3_cache_smoke' "${ROOT}/docs/gpu-profiles-l3.md"
grep -q 'l3_spec_cache_smoke' "${ROOT}/docs/gpu-profiles-l3.md"
grep -q 'l3-cuda-full-gate' "${ROOT}/docs/gpu-profiles-l3.md"
grep -q 'RUN_E2E_L3' "$(spath gpu_5080_session.sh)"
grep -q 'prompt_cache_keys' "${ROOT}/runtime/runtime/cache_bridge.py"
grep -q 'IsMLX()' "${ROOT}/docs/mlx-routing-policy.md"
grep -q 'modelUsesRuntimeInference' "${ROOT}/server/runtime_inference_routing.go"
grep -q 'RUN_E2E_LEGACY=1 with RUN_E2E_GPU' "$(spath e2e_runtime_smoke.sh)"
grep -q 'RUN_E2E_VRAM_CLAMP' "$(spath gpu_clamp_smoke.sh)"
grep -q 'phase12_capture_tool_transcript' "$(spath phase12_capture_tool_transcript.sh)" || grep -q 'X-Zerollama-Runtime' "$(spath phase12_capture_tool_transcript.sh)"
grep -q 'RUN_E2E_PHASE14' "$(spath e2e_runtime_smoke.sh)"
grep -q 'smoke_runtime_needs_server_bin' "$(spath runtime_smoke_lib.sh)"
grep -q 'internal/tokenize' "$(spath e2e_runtime_smoke.sh)"
grep -q 'smoke_llama_model_config_hint' "$(spath runtime_smoke_lib.sh)"
grep -q 'truncate_mode=tokenize' "$(spath e2e_runtime_smoke.sh)"
grep -q 'smoke_runtime_require_phase14_endpoints' "$(spath runtime_smoke_lib.sh)"
grep -q 'smoke_runtime_assert_llama_backend_source' "$(spath runtime_smoke_lib.sh)"
grep -q 'skip_global_vram_factor_export' "${ROOT}/runtime/runtime/gpu_health_report.py"
grep -q 'skip_global_vram_factor_export' "${ROOT}/runtime/runtime/gpu_snapshot.py"
grep -q 'vram_recommendations' "${ROOT}/runtime/runtime/gpu_health_report.py"
grep -q 'skip_global_vram_factor_export' "${ROOT}/runtime/tests/test_vram_recommendations.py"
grep -q 'smoke_runtime_apply_backend_flags_from_health' "$(spath runtime_smoke_lib.sh)"
grep -q 'inferred from /health' "$(spath phase14_yaml_config_smoke.sh)"
grep -q 'RUN_E2E_LLAMA_BACKEND_SOURCE=env' "$(spath phase14_inprocess_smoke.sh)"
grep -q 'RUN_E2E_INPROCESS=1' "$(spath phase14_inprocess_smoke.sh)"
grep -q 'RUN_E2E_LLAMA_CPP_PYTHON=1' "$(spath phase14_wheel_cpu_smoke.sh)"
grep -q 'RUN_E2E_LLAMA_BACKEND_SOURCE=env' "$(spath phase14_wheel_cpu_smoke.sh)"
grep -q 'RUN_E2E_LLAMA_BACKEND_SOURCE=config' "$(spath phase14_yaml_config_smoke.sh)"
grep -q 'RUN_E2E_LLAMA_BACKEND_SOURCE=default' "$(spath phase14_subprocess_default_smoke.sh)"
grep -q 'canonical_llama_backend' "${ROOT}/runtime/runtime/worker/factory.py"
grep -q 'llama_backend_from_file' "${ROOT}/runtime/runtime/config.py"
grep -q 'RUN_E2E_PHASE14_SIGNOFF' "$(spath gpu_5080_session.sh)"
grep -q 'RUN_E2E_PHASE15' "$(spath gpu_5080_session.sh)"
grep -q '_saved_phase14_signoff' "$(spath gpu_5080_session.sh)"
grep -q 'phase14_5080_signoff.sh' "$(spath gpu_smoke_all.sh)"
grep -q 'phase15_inprocess_signoff.sh' "$(spath gpu_smoke_all.sh)"
grep -q 'phase15_inprocess_kv_smoke.sh' "$(spath phase15_inprocess_signoff.sh)"
grep -q 'phase15_inprocess_multiseq_smoke.sh' "$(spath phase15_inprocess_signoff.sh)"
grep -q 'RUN_E2E_PHASE14' "$(spath gpu_smoke_all.sh)"
grep -q 'llama_cpp_wheel_health' "${ROOT}/runtime/runtime/worker/llama_cpp_python.py"
grep -q 'llama_cpp' "$(spath gpu_phase13_snapshot.sh)"
grep -q 'smoke_runtime_assert_llama_cpp_gpu' "$(spath runtime_smoke_lib.sh)"
grep -q 'RUN_E2E_LLAMA_CPP_PYTHON_GPU=1' "$(spath phase14_wheel_gpu_smoke.sh)"
grep -q 'phase14_inprocess_smoke.sh' "$(spath phase14_both_backends.sh)"
grep -q 'phase14_wheel_cpu_smoke.sh' "$(spath phase14_both_backends.sh)"
grep -q 'checkLoopbackPortFree' "${ROOT}/x/runtimeworker/client.go"
grep -q 'embed_boot' "${ROOT}/runtime/runtime/engine.py"
grep -q 'kv_decode_steps (generate)' "$(spath e2e_runtime_smoke.sh)"
grep -q 'phase14_yaml_config_full_smoke.sh' "$(spath phase14_5080_signoff.sh)"
grep -q 'phase15_inprocess_signoff.sh' "$(spath phase14_5080_signoff.sh)"
grep -q 'phase14_both_backends.sh' "$(spath phase14_5080_signoff.sh)"
grep -q 'phase15_inprocess_kv_smoke.sh' "$(spath check_gpu_scripts.sh)"
grep -q 'phase14_backend_smoke.sh' "$(spath phase15_inprocess_kv_smoke.sh)"
grep -q 'smoke_runtime_assert_kv_snapshot' "$(spath runtime_smoke_lib.sh)"
grep -q 'smoke_runtime_assert_kv_snapshot' "$(spath phase15_inprocess_kv_smoke.sh)"
grep -q 'smoke_runtime_assert_kv_snapshot' "$(spath phase15_inprocess_multiseq_smoke.sh)"
grep -q 'llama_parallel_slots: 2' "$(spath phase15_inprocess_multiseq_smoke.sh)"
grep -q 'ZEROLLAMA_RUNTIME_CONFIG' "$(spath phase14_yaml_config_full_smoke.sh)"
grep -q 'phase14_yaml_config_smoke.sh' "$(spath phase14_yaml_config_full_smoke.sh)"
grep -q '/api/train/status' "$(spath e2e_training_ops_smoke.sh)"
grep -q 'cmd.*ping' "$(spath e2e_training_ops_smoke.sh)"
grep -q 'ZEROLLAMA_RUNTIME_SHARED_PYTHON' "${ROOT}/runtime/runtime/env.py"
grep -q 'health try' "$(spath repro_shared_interpreter_health_hang.sh)"
grep -q 'llama_patch_doctor' "$(spath phase15_kv_native_ci.sh)"
grep -q 'phase15_llama_kv_ext_pin_check' "$(spath phase15_kv_native_ci.sh)"
grep -q 'test_kv_native_parity' "$(spath phase15_kv_native_ci.sh)"
grep -q 'test_kv_decode_long_ctx' "$(spath phase15_kv_native_ci.sh)"
grep -q 'ZEROLLAMA_KV_DECODE_LOOP' "$(spath phase15_kv_native_ci.sh)"
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
grep -q 'phase15_health_smoke' "$(spath phase15_kv_native_ci.sh)"
grep -q 'record_decode_step' "${ROOT}/runtime/runtime/worker/libllama_ctypes.py"
grep -q 'kv_slot' "${ROOT}/runtime/runtime/scheduler/scheduler.py"
grep -q 'resolve_parallel_slots' "${ROOT}/runtime/runtime/llama_args.py"
grep -q '_effective_llama_parallel_slots' "${ROOT}/runtime/runtime/engine.py"
grep -q 'P17_SERVE_EXTRA' "$(spath phase17_llama_server_smoke.sh)"
grep -q 'P17_ASSERT_RUNTIME_OFF' "$(spath phase17_llama_server_smoke.sh)"
grep -q 'phase17_llama_server_smoke.sh' "$(spath phase16_edge_smoke.sh)"
grep -q 'ApplyServeBackendEnv' "${ROOT}/cmd/cmd.go"
grep -q 'LinuxLlamaServerAutoEnv' "${ROOT}/envconfig/serve_backend.go"
grep -q 'EdgeMode()' "${ROOT}/envconfig/config.go"
grep -q 'ZEROLLAMA_RUNTIME_EMBED' "${ROOT}/envconfig/config.go"
grep -q 'build_zerollama_edge.sh' "${ROOT}/docs/phase16-thin-edge.md"
grep -q 'RUN_E2E_EDGE' "$(spath gpu_5080_session.sh)"
grep -q 'RUN_E2E_P17' "$(spath gpu_5080_session.sh)"
grep -q 'doctorCheckEdgeBuild' "${ROOT}/cmd/doctor.go"
grep -q 'BackendPolicy' "${ROOT}/api/types.go"
grep -q 'P17_LINUX_AUTO' "$(spath phase17_llama_server_smoke.sh)"
grep -q 'RUN_E2E_P17_LINUX_AUTO' "$(spath gpu_5080_session.sh)"
grep -q 'RUN_E2E_UPSTREAM_GGUF' "$(spath gpu_5080_session.sh)"
grep -q 'upstream bundle: restart serve' "$(spath gpu_5080_session.sh)"
grep -q 'serve_production_wrapper' "$(spath serve_production_wrapper.sh)"
grep -q 'WHY this wrapper exists' "$(spath serve_production_wrapper.sh)"
grep -q 'cannot find zerollama repo' "$(spath serve_gpu_example.sh)"
grep -q 'RUN_E2E_P17_VISION' "$(spath gpu_5080_session.sh)"
grep -q 'serve_linux_auto.sh' "${ROOT}/docs/phase17-llama-server.md"
grep -q 'P15_PIN_JSON' "$(spath phase15_llama_kv_ext_pin_check.sh)"
grep -q 'phase16_edge_binary_smoke.sh' "$(spath check_gpu_scripts.sh)"
grep -q 'phase16_edge_build_smoke.sh' "${ROOT}/.github/workflows/zerollama-regression.yaml"
grep -q 'run_upstream_gguf' "${ROOT}/.github/workflows/zerollama-gpu-smoke.yaml"
grep -q 'GgmlRunnerLinked' "${ROOT}/envconfig/ggml_runner.go"
grep -q 'phase15_llama_kv_ext_pin_check.sh' "${ROOT}/.github/workflows/zerollama-regression.yaml"
grep -q 'doctorCheckLlamaPatches' "${ROOT}/cmd/doctor.go"
grep -q 'llama_patch_doctor.sh' "${ROOT}/.github/workflows/zerollama-regression.yaml"
grep -q 'doctorCheckGgmlRunner' "${ROOT}/cmd/doctor.go"
grep -q 'schedSkipGgmlRunnerLoad' "${ROOT}/server/sched.go"
grep -q 'phase17_l2_pin_status.sh' "${ROOT}/.github/workflows/zerollama-regression.yaml"
grep -q 'phase15_upstream_kv_watch.sh' "${ROOT}/.github/workflows/zerollama-regression.yaml"
grep -q 'Operator troubleshooting' "${ROOT}/docs/phase17-llama-server.md"
grep -q '5080_build_vendor_llama_server' "$(spath 5080_env.sh)"
grep -q '5080_resignoff' "${ROOT}/docs/5080-runbook.md"
grep -q 'RUN_E2E_L3_RADIX' "$(spath gpu_5080_session.sh)"
grep -q 'L3_RUN_RADIX' "$(spath l3_cuda_full_gate.sh)"
grep -q 'CUDA_LLAMA_MODEL' "$(spath l3_radix_prefix_smoke.sh)"
grep -q 'e2e_t6_queue_smoke.sh' "$(spath check_gpu_scripts.sh)"
grep -q 'inference.training.queue_policy' "$(spath e2e_t6_queue_smoke.sh)"
grep -q 'TrainingQueuePolicy' "${ROOT}/api/types.go"
grep -q 't6-unified-queue.md' "${ROOT}/docs/README.md"
echo "ok: e2e_runtime_smoke tools markers"

echo "PASS: check_gpu_scripts"
