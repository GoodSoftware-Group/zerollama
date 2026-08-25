#!/usr/bin/env bash
# Register LTXV config-only manifests (no GGUF weights in Ollama blobs).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
GO="${GO:-go}"
register() {
  local name="$1"
  local cfg="$2"
  "$GO" run ./scripts/register_wan_manifest "$name" "$cfg"
}
register ltxv-13b-distilled:16g modelfiles/ltxv-13b-distilled/config.json
