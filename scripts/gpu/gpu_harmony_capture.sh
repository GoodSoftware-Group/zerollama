#!/usr/bin/env bash
# Capture gpt-oss (harmony parser) tools transcript on a GPU host (5080-class).
#
#   export OLLAMA_HOST=http://127.0.0.1:8080
#   ./scripts/gpu/gpu_harmony_capture.sh
#   ./scripts/gpu/gpu_harmony_capture.sh --out /tmp/harmony-capture.json
#
# Do not set RUN_E2E_GGUF / LLAMA_MODEL for this script — uses pulled model weights.
# gpt-oss:20b (MXFP4) may require ~40+ GiB *host RAM* for mmap on runtime path — NOT required
# on 5080-class ~19 GiB hosts; CI uses TestGoldenHarmonyParseToolOutput instead.
# WHY API unload (runtime_smoke_lib): same as other GPU smokes — no pkill on ggml runners.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${GPU_HARMONY_MODEL:-gpt-oss:20b}"
OUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,9p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:8080}"
export ZEROLLAMA_RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"

# shellcheck source=scripts/runtime/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime/runtime_smoke_lib.sh"

echo "== unload stale ggml runners (optional) =="
smoke_unload_ggml_runners || {
  echo "warn: could not unload via API; try RUN_E2E_UNLOAD_MODEL=<loaded tag>" >&2
}

unset RUN_E2E_GGUF
runtime_resume_if_needed
smoke_prepare_vram_for_runtime
runtime_resume_if_needed

args=(--harmony --validate-parse)
if [[ -n "$OUT" ]]; then
  args+=(--out "$OUT")
fi

curl -sf -m 30 "${ZEROLLAMA_RUNTIME_URL%/}/health" >/dev/null || {
  echo "runtime /health failed" >&2
  exit 1
}

echo "== capture harmony tools ($MODEL) =="
echo "note: runtime path checks host RAM at load (gpt-oss:20b MXFP4 often needs 32+ GiB RAM)" >&2
if ! "${ROOT}/scripts/phase/phase12_capture_tool_transcript.sh" "$MODEL" "${args[@]}"; then
  echo "hint: gpt-oss:20b often needs >32 GiB host RAM on runtime path; try more RAM or a smaller quant/model" >&2
  exit 1
fi

echo "PASS: gpu_harmony_capture"
