#!/usr/bin/env bash
# Register stable-diffusion.cpp image presets (config-only manifests).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GO="${GO:-go}"
register() {
  local name="$1"
  local cfg="$2"
  echo "registering $name ..."
  (cd "$REPO_ROOT" && "$GO" run -mod=mod ./scripts/register_wan_manifest "$name" "$cfg")
}
for name in sd15-vulkan sd15-q8-vulkan sd15-turbo-vulkan sdxl-vulkan; do
  register "$name" "$REPO_ROOT/modelfiles/$name/config.json"
done
echo "done. OLLAMA_EXTERNAL_IMAGE_BIN=/usr/lib/ollama-zerollama/sd_external_image.sh"
