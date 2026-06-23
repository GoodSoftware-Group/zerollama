#!/usr/bin/env bash
# SGLang xfer operator gate — video expansion caches (unit) + optional L3 text smoke.
#
# WHY: Video session cache (ffmpeg/URL) and L3 prefix KV (cached_tokens) are separate
# layers. CI proves video wiring without GPU; operators opt into L3 with the same
# prompt_cache_key discipline used for repeat clips.
#
# Usage:
#   ./scripts/video_l3_agent_gate.sh
#
# Optional L3 wiring (subprocess + GGUF; no VLM required):
#   RUN_E2E_L3=1 CUDA_LLAMA_MODEL=/path/to/model.gguf ./scripts/video_l3_agent_gate.sh
#
# Optional live video expand (ffmpeg + VLM; no L3 cached_tokens assertion):
#   RUN_E2E_VIDEO_AGENT=1 VIDEO_SMOKE_MODEL=qwen3-vl:latest OLLAMA_HOST=... \
#     ./scripts/video_l3_agent_gate.sh
#
# Optional live video + full inference + cached_tokens (GPU + L3):
#   RUN_E2E_VIDEO_AGENT_INFER=1 VIDEO_SMOKE_MODEL=qwen3-vl:latest \
#     OLLAMA_HOST=http://127.0.0.1:8080 VIDEO_AGENT_GO_LOG=/tmp/zerollama-go.log \
#     ./scripts/video_l3_agent_gate.sh
#
# Env:
#   L3_OUT — passed through to l3_cache_smoke (default /tmp/l3-cache-smoke.json)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "== video agent unit gate =="
"${ROOT}/scripts/video_agent_cache_smoke.sh"

if [[ "${RUN_E2E_VIDEO_AGENT:-0}" == "1" ]]; then
  echo "== live video agent (expand cache) already ran inside video_agent_cache_smoke =="
fi

if [[ "${RUN_E2E_L3:-0}" == "1" ]]; then
  echo "== L3 prefix cache smoke (text; same prompt_cache_key discipline) =="
  L3_OUT="${L3_OUT:-/tmp/l3-cache-smoke.json}"
  export L3_OUT
  "${ROOT}/scripts/l3_cache_smoke.sh"
  "${ROOT}/scripts/l3_gate_report.sh" "${L3_OUT}"
fi

if [[ "${RUN_E2E_VIDEO_AGENT_INFER:-0}" == "1" ]]; then
  echo "== live video + inference cached_tokens gate =="
  "${ROOT}/scripts/video_agent_infer_smoke.sh"
fi

echo "PASS video_l3_agent_gate"
echo "Doc: docs/sglang-multimodal-borrowings.md docs/gpu-profiles-l3.md"
