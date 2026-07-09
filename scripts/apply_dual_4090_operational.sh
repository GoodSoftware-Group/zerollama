#!/usr/bin/env bash
# Apply dual-4090 operational fixes: runtime code, YAML, minimal systemd drop-in, restart.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL="${Z4090_INSTALL:-/opt/zerollama}"
RT="${INSTALL}/runtime"

echo ">>> sync runtime (gpu_profiles, engine, dual_4090.yaml)"
sudo cp "${ROOT}/runtime/runtime/gpu_profiles.py" "${RT}/runtime/gpu_profiles.py"
sudo cp "${ROOT}/runtime/runtime/engine.py" "${RT}/runtime/engine.py"
sudo cp "${ROOT}/runtime/runtime/vram_yaml_defaults.py" "${RT}/runtime/vram_yaml_defaults.py"

echo ">>> install minimal ollama.service drop-in (serve defaults from YAML)"
sudo mkdir -p /etc/systemd/system/ollama.service.d
# Remove legacy overrides that fought autoconfig (MAX_LOADED=3, KEEP_ALIVE=-1).
for f in override.conf zerollama.conf dual_4090.conf; do
  if [[ -f "/etc/systemd/system/ollama.service.d/${f}" ]]; then
    sudo rm -f "/etc/systemd/system/ollama.service.d/${f}"
  fi
done
sudo cp "${ROOT}/scripts/systemd/dual_4090-ollama.conf" \
  /etc/systemd/system/ollama.service.d/dual_4090.conf

sudo cp "${ROOT}/runtime/configs/dual_4090.yaml" "${RT}/configs/dual_4090.yaml"

if [[ -f /tmp/zerollama-edge ]]; then
  echo ">>> install rebuilt zerollama (hardware_lane autoconfig)"
  sudo cp /tmp/zerollama-edge /usr/local/bin/ollama
fi

echo ">>> restart runtime + ollama"
sudo systemctl daemon-reload
sudo systemctl restart zerollama-runtime.service
sleep 3
sudo systemctl restart ollama.service
sleep 2

echo ">>> health"
curl -sf http://127.0.0.1:8081/ready | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('runtime ready:', d.get('ready'), d.get('ready_reasons'))
gp = d.get('gpu_profile') or {}
print('n_parallel:', gp.get('n_parallel'))
"
curl -sf http://127.0.0.1:2083/api/ps | python3 -c "import json,sys; print('loaded:', [m['name'] for m in json.load(sys.stdin).get('models',[])])"

echo ">>> done — serve defaults now come from dual_4090.yaml serve: block when OLLAMA_* unset"
