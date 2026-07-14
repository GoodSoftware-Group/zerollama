#!/usr/bin/env bash
# Register OpenVINO GenAI image presets (config-only manifests).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GO="${GO:-go}"
register() {
  local name="$1"
  local cfg="$2"
  echo "registering $name ..."
  (cd "$REPO_ROOT" && "$GO" run -mod=mod ./scripts/register_wan_manifest "$name" "$cfg")
}
for name in sd15-openvino sdxl-openvino; do
  register "$name" "$REPO_ROOT/modelfiles/$name/config.json"
done
echo "done. Vulkan SD models keep OLLAMA_EXTERNAL_IMAGE_BIN=sd_external_image.sh; OV models use per-manifest ov_external_image.sh"
