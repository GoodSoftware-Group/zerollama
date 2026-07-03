#!/usr/bin/env bash
# Intel Arc A380 sign-off driver — Vulkan ggml inference path.
#
# Usage:
#   cd ~/zerollama
#   source ./scripts/a380_env.sh
#   ./scripts/a380_signoff.sh
#   ./scripts/a380_signoff.sh --build
#   ./scripts/a380_signoff.sh --tier 1
#
# Research baseline: ~/bmtl/asm_lab/lanes/arc-a380/runs/research_synthesis.json
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/a380_env.sh
source "${ROOT}/scripts/a380_env.sh"

TIER="${A380_SIGNOFF_TIER:-all}"
DO_BUILD=0
NO_SERVE=0

usage() {
  sed -n '2,11p' "$0" | tail -n +2
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage 0 ;;
    --tier) TIER="${2:?}"; shift 2 ;;
    --build) DO_BUILD=1; shift ;;
    --no-serve) NO_SERVE=1; shift ;;
    *) echo "unknown arg: $1" >&2; usage 1 ;;
  esac
done

run_tier() {
  local want="$1"
  [[ "${TIER}" == "all" || "${TIER}" == "${want}" ]]
}

echo "== a380 sign-off (tier=${TIER}) =="
a380_print_env

if [[ -f "${ZA380_RESEARCH_SYNTHESIS}" ]]; then
  echo "== research synthesis (reference) =="
  if command -v jq >/dev/null 2>&1; then
    jq -r '.lane_verdict, .recommended_defaults.intel_gpu, .ollama_decode_scaling.production_guidance' \
      "${ZA380_RESEARCH_SYNTHESIS}" 2>/dev/null || true
  else
    python3 -c "import json; d=json.load(open('${ZA380_RESEARCH_SYNTHESIS}')); print(d.get('lane_verdict','')); print(d.get('recommended_defaults',{}).get('intel_gpu','')); print(d.get('ollama_decode_scaling',{}).get('production_guidance',''))"
  fi
fi

if [[ "${DO_BUILD}" -eq 1 ]]; then
  echo "== build zerollama + vendor llama-server =="
  a380_build_zerollama
fi

if run_tier 0; then
  echo "== tier 0: vulkan + profile JSON + fork llama-server =="
  a380_check_vulkan
  if [[ -n "${LLAMA_SERVER_BIN:-}" ]]; then
    a380_verify_fork_llama_server "${LLAMA_SERVER_BIN}"
  elif ! a380_export_llama_vendor_env || ! a380_verify_fork_llama_server; then
    echo "a380: tier 0 FAIL — install vendor llama-server:" >&2
    echo "  ./scripts/build_zerollama_a380.sh && sudo ./scripts/install_a380_llama_server.sh" >&2
    exit 1
  fi
  (cd "${ROOT}/runtime" && python3 - <<PY
from runtime.gpu_profiles import load_gpu_config, flags_from_gpu_config
cfg = load_gpu_config("arc-a380")
flags, _ = flags_from_gpu_config(cfg, fork_enabled=False)
assert flags.get("n_parallel") == 1
assert flags.get("ctx_size") == 4096
print("arc-a380 profile OK", flags.get("batch_size"))
PY
  )
fi

if [[ "${NO_SERVE}" -eq 0 ]]; then
  echo "== start serve =="
  a380_start_serve
else
  export A380_ASSUME_SERVE_UP=1
fi

if run_tier 1; then
  echo "== tier 1: gpu profile unit (forced id) =="
  (cd "${ROOT}/runtime" && ZEROLLAMA_GPU_PROFILE_ID=arc-a380 python3 - <<PY
import os
from pathlib import Path
from runtime.config import RuntimeConfig
os.environ["ZEROLLAMA_GPU_PROFILE"] = "1"
cfg = RuntimeConfig.from_file(Path("configs/arc_a380.yaml"))
assert cfg.gpu_profile is not None
assert cfg.gpu_profile["id"] == "arc-a380"
print("runtime profile", cfg.gpu_profile["id"], "source", cfg.gpu_profile.get("source"))
PY
  )
fi

if run_tier 2; then
  echo "== tier 2: vulkan API smoke =="
  "${ROOT}/scripts/a380_vulkan_smoke.sh"
fi

echo "== a380 sign-off complete =="
echo "Report total_duration_eval_tok_s at your decode length — not eval_tok_s alone."
echo "See docs/a380-runbook.md and ${ZA380_RESEARCH_LANE}/README.md"
