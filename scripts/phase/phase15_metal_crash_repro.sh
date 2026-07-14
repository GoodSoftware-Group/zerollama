#!/usr/bin/env bash
# Bisect intermittent Metal / native-KV crashes with tracing ON and workarounds OFF.
#
# Prerequisites (same as phase15_metal_signoff):
#   - zerollama serve on :8080/:8081 with inprocess backend + Metal libllama
#   - LLAMA_MODEL or RUN_E2E_GGUF set to a small local GGUF
#   - Linked _kv_native ext built (phase15_kv_decode_loop_build.sh)
#
# Usage:
#   export ZEROLLAMA_INFER_TRACE=1          # required for phase logs
#   export ZEROLLAMA_KV_NATIVE_DECODE=1     # default; do not disable for bisect
#   export ZEROLLAMA_KV_NATIVE_SAMPLE=1
#   bash scripts/phase/phase15_metal_crash_repro.sh [scenario]
#
# Scenarios (run one or all):
#   runtime_loop   — 5× direct /internal/generate on :8081 (no broker)
#   broker_gguf    — broker + generate with RUN_E2E_GGUF still set (reload churn)
#   phase14_full   — full phase14_backend_smoke without unset RUN_E2E_GGUF
#
# On crash:
#   1. tail -200 /tmp/macos-runtime.log | rg 'infer_trace|create_tensor|GGML_ASSERT'
#   2. ls -lt ~/Library/Logs/DiagnosticReports/Python*.ips | head -3
#   3. Re-run under lldb (see docs below) with GGML_BACKTRACE_LLDB=1
#
# Attach lldb to a running sidecar (find PID from macos-runtime.log or ps):
#   export GGML_BACKTRACE_LLDB=1
#   lldb -p "$(pgrep -f 'runtime.server.app' | head -1)" \
#     -o 'process handle SIGSEGV -n true -p true -s false' \
#     -o 'continue'
#   # then run a failing scenario in another terminal
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/runtime/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime/runtime_smoke_lib.sh"

export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:8080}"
RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"
export ZEROLLAMA_INFER_TRACE="${ZEROLLAMA_INFER_TRACE:-1}"
export ZEROLLAMA_KV_NATIVE_DECODE="${ZEROLLAMA_KV_NATIVE_DECODE:-1}"
export ZEROLLAMA_KV_NATIVE_SAMPLE="${ZEROLLAMA_KV_NATIVE_SAMPLE:-1}"

if [[ -z "${LLAMA_MODEL:-}" && -z "${RUN_E2E_GGUF:-}" ]]; then
  echo "Set LLAMA_MODEL or RUN_E2E_GGUF" >&2
  exit 1
fi

_gguf="${RUN_E2E_GGUF:-${LLAMA_MODEL:-}}"
_scenario="${1:-all}"

echo "== phase15_metal_crash_repro =="
echo "infer_trace=${ZEROLLAMA_INFER_TRACE} native_decode=${ZEROLLAMA_KV_NATIVE_DECODE} native_sample=${ZEROLLAMA_KV_NATIVE_SAMPLE}"
echo "runtime=${RUNTIME_URL} gguf=${_gguf} scenario=${_scenario}"

smoke_runtime_require_listening "$RUNTIME_URL"
_health=$(runtime_fetch_health "$RUNTIME_URL")
echo "/health llama_backend=$(smoke_runtime_llama_backend "$_health" strict)"

_runtime_generate() {
  local label="$1"
  local n="${2:-8}"
  echo "-- generate (${label}) num_predict=${n} --"
  python3 - <<PY
import json, os, urllib.request
url = os.environ["RUNTIME_URL"] + "/api/generate"
payload = {
    "model": "smoke",
    "prompt": "Say one short word.",
    "stream": False,
    "options": {
        "num_predict": ${n},
        "num_ctx": 4096,
        "temperature": 0.7,
        "gguf": os.environ["GGUF"],
    },
}
req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=300) as r:
    body = json.loads(r.read().decode())
print("response_len", len(body.get("response") or ""))
PY
}

scenario_runtime_loop() {
  export RUNTIME_URL GGUF="$_gguf"
  for i in 1 2 3 4 5; do
    _runtime_generate "loop_${i}" 8 || { echo "FAIL loop ${i}"; return 1; }
  done
  echo "PASS scenario_runtime_loop"
}

scenario_broker_gguf() {
  export RUN_E2E_GGUF="$_gguf"
  smoke_prepare_vram_for_runtime
  export RUNTIME_URL GGUF="$_gguf"
  _runtime_generate "after_broker_with_gguf" 8
  echo "PASS scenario_broker_gguf"
}

scenario_phase14_full() {
  export RUN_E2E_GGUF="$_gguf"
  export RUN_E2E_GPU=1 RUN_E2E_PHASE14=1 RUN_E2E_PROXY=0
  [[ "${RUN_E2E_INPROCESS:-0}" == "1" ]] && export RUN_E2E_INPROCESS=1
  smoke_prepare_vram_for_runtime
  # Intentionally keep RUN_E2E_GGUF set (workaround disabled) to repro reload.
  env RUN_E2E_GGUF="$_gguf" RUN_E2E_GPU=1 RUN_E2E_PHASE14=1 RUN_E2E_PROXY=0 \
    "${ROOT}/scripts/e2e/e2e_runtime_smoke.sh"
  echo "PASS scenario_phase14_full"
}

_run_one() {
  case "$1" in
    runtime_loop) scenario_runtime_loop ;;
    broker_gguf) scenario_broker_gguf ;;
    phase14_full) scenario_phase14_full ;;
    *) echo "unknown scenario: $1 (runtime_loop|broker_gguf|phase14_full|all)" >&2; return 1 ;;
  esac
}

case "$_scenario" in
  all)
    _run_one runtime_loop
    _run_one broker_gguf
    _run_one phase14_full
    ;;
  *)
    _run_one "$_scenario"
    ;;
esac

echo "PASS phase15_metal_crash_repro (${_scenario})"
