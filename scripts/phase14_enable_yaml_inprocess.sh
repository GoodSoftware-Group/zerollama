#!/usr/bin/env bash
# After phase14_backend_smoke PASS with inprocess: enable packaged YAML default on 5080-class hosts.
#
# WHY a script: uncommenting llama_backend: inprocess is the last Phase 14 operator step; this
# avoids hand-editing and prints restart instructions.
#
#   RUN_E2E_INPROCESS=1 ./scripts/phase14_backend_smoke.sh   # must pass first
#   ./scripts/phase14_enable_yaml_inprocess.sh
#   # restart serve WITHOUT ZEROLLAMA_RUNTIME_LLAMA_BACKEND
#   RUN_E2E_LLAMA_BACKEND_SOURCE=config ./scripts/phase14_yaml_config_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YAML="${ZEROLLAMA_RUNTIME_CONFIG:-${ROOT}/runtime/configs/single_gpu.yaml}"

if [[ ! -f "$YAML" ]]; then
  echo "Config not found: $YAML" >&2
  exit 1
fi

if grep -q '^llama_backend: inprocess' "$YAML"; then
  echo "Already enabled: llama_backend: inprocess in $YAML"
  exit 0
fi

if ! grep -q '^# llama_backend: inprocess' "$YAML"; then
  echo "Expected commented line '# llama_backend: inprocess' in $YAML" >&2
  exit 1
fi

sed -i 's/^# llama_backend: inprocess/llama_backend: inprocess/' "$YAML"
echo "Enabled llama_backend: inprocess in $YAML"
echo ""
echo "Next:"
echo "  1. Restart serve without ZEROLLAMA_RUNTIME_LLAMA_BACKEND"
echo "  2. RUN_E2E_LLAMA_BACKEND_SOURCE=config ./scripts/phase14_yaml_config_smoke.sh"
