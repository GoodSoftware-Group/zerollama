#!/usr/bin/env bash
# Smoke Go↔runtime coordination mirrors (Phase 11). No inference job submit.
#
#   OLLAMA_HOST=http://127.0.0.1:8080 ZEROLLAMA_RUNTIME_URL=http://127.0.0.1:8081 ./scripts/e2e/e2e_coordination_smoke.sh
#
# Requires zerollama serve with embedded or external runtime on :8081.
set -euo pipefail

OLLAMA_URL="${OLLAMA_HOST:-http://127.0.0.1:8080}"
RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"
TIMEOUT="${E2E_COORD_TIMEOUT:-10}"

health_json=$(curl -sf -m "$TIMEOUT" "${RUNTIME_URL}/health")
python3 -c "
import json, sys
h = json.loads(sys.argv[1])
ad = h.get('admission') or {}
coord = (ad.get('backpressure') or {}).get('coordination') or {}
if not ad.get('inference_policy', True):
    print('warn: inference_policy off')
fresh = coord.get('fresh')
if fresh is False and coord.get('stale') is True:
    print('warn: go_coordination mirror stale (push may be failing)')
elif fresh:
    print('go_coordination: fresh')
else:
    print('go_coordination: unknown (embedded-only runtime without Go pusher is ok)')
for k in ('shared_interpreter', 'vram_probe_effective', 'go_training_gpu_busy'):
    if k in h or k in ad:
        print(f'{k}:', ad.get(k, h.get(k)))
gates = dict(ad.get('gates_active') or {})
compat = ad.get('gates_active_compat') or {}
if gates and 'low_would_wait' not in gates and 'batch_backpressure' in gates:
    print('note: runtime lacks gates_active v2 keys — rebuild/restart zerollama')
if gates:
    on = {k: gates[k] for k in sorted(gates) if gates.get(k)}
    if on:
        print('gates_active (true => throttles priority=low only):', on)
if compat:
    on = {k: compat[k] for k in sorted(compat) if compat.get(k)}
    if on and on != {k: gates[k] for k in on if k in gates}:
        print('gates_active_compat:', on)
vb = h.get('vram_budget') or {}
if vb.get('admission_fits') is not None:
    print('vram_budget.admission_fits:', vb.get('admission_fits'))
suggest = vb.get('suggested_max_num_ctx')
if suggest is not None:
    print('vram_budget.suggested_max_num_ctx:', suggest)
    if vb.get('num_ctx_over_budget'):
        print('vram_budget.num_ctx_over_budget:', vb.get('num_ctx_over_budget'))
ac = h.get('autoconfig') or {}
if ac:
    print('autoconfig:', ac.get('pick'), ac.get('config_path'))
policy = h.get('vram_num_ctx_policy') or {}
if policy:
    print('vram_num_ctx_policy:', policy.get('env'), 'clamp=', policy.get('clamp_enabled'))
if ad.get('vram_min_free_configured') is not None:
    print('admission.vram_min_free_configured:', ad.get('vram_min_free_configured'))
if ad.get('vram_training_reserve_configured') is not None:
    print('admission.vram_training_reserve_configured:', ad.get('vram_training_reserve_configured'))
print('admission backlog:', ad.get('backlog', h.get('model_swap')))
" "$health_json"

# Optional: training status when enabled
if curl -sf -m 2 "${OLLAMA_URL}/api/train/status" >/dev/null 2>&1; then
  echo "train/status: ok"
else
  echo "train/status: skipped (training disabled or 404)"
fi

echo "PASS: coordination smoke"
