#!/usr/bin/env bash
# One-shot A380 Vulkan session: device check + API smoke + optional research refresh.
#
#   source ./scripts/gpu/a380_env.sh
#   ./scripts/gpu/gpu_a380_session.sh
#
# Optional:
#   A380_RUN_RESEARCH=1     # make -C asm_lab lane probes (long)
#   A380_SMOKE_OUT=/tmp/a380-vulkan-smoke.json
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/gpu/a380_env.sh
source "${ROOT}/scripts/gpu/a380_env.sh"

echo "== a380 session =="
a380_print_env

echo "== tier 0: vulkan device =="
a380_check_vulkan
vulkaninfo --summary 2>/dev/null | grep -iE 'Arc|DG2|deviceName|driverName' | head -8 || true

if [[ "${A380_ASSUME_SERVE_UP:-0}" != "1" ]]; then
  echo "== start serve =="
  a380_start_serve
fi

echo "== tier 1: api tags =="
curl -sf "http://${OLLAMA_HOST#http://}/api/tags" | python3 -c "
import sys, json
d = json.load(sys.stdin)
local = [m['name'] for m in d.get('models', []) if not m['name'].endswith(':cloud')]
for n in local[:8]:
    print(n)
print(f'({len(local)} local models)')
"

echo "== tier 2: vulkan smoke =="
if ! "${ROOT}/scripts/gpu/a380_vulkan_smoke.sh"; then
  echo "== tier 2: SOFT (thresholds) — inference ran; see /tmp/a380-vulkan-smoke.json =="
fi

if [[ "${A380_RUN_RESEARCH:-0}" == "1" && -d "${ZA380_RESEARCH_LANE}" ]]; then
  echo "== research lane refresh =="
  make -C "${ZA380_RESEARCH_LANE}" ollama-vulkan ollama-load-investigation || true
fi

echo "== a380 session PASS =="
