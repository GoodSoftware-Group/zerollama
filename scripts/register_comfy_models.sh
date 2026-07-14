#!/usr/bin/env bash
# Register config-only ComfyUI image manifests (no weights in Ollama blobs; see
# docs/comfyui-image-backend.md). With no args, registers the full default set.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
GO="${GO:-go}"

register() {
  local name="$1"
  local cfg="$2"
  "$GO" run ./scripts/register_wan_manifest "$name" "$cfg"
}

if [ "$#" -eq 2 ]; then
  register "$1" "$2"
  exit 0
fi

if [ "$#" -ne 0 ]; then
  echo "usage: $0 [<model-name> <config.json>]" >&2
  exit 2
fi

register comfy/qwen-image modelfiles/comfy-qwen-image/config.json
register comfy/qwen-image-edit modelfiles/comfy-qwen-image-edit/config.json
register comfy/flux1-dev modelfiles/comfy-flux1-dev/config.json
register comfy/flux2-dev modelfiles/comfy-flux2-dev/config.json
register comfy/glm-image modelfiles/comfy-glm-image/config.json
register comfy/flux2-klein-9b modelfiles/comfy-flux2-klein-9b/config.json
