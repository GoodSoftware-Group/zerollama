#!/usr/bin/env bash
# Register Wan config-only manifests (no GGUF weights in Ollama blobs).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
GO="${GO:-go}"
register() {
  local name="$1"
  local cfg="$2"
  "$GO" run ./scripts/register_wan_manifest "$name" "$cfg"
}
register wan2.1-t2v:1.3b modelfiles/wan2.1-t2v/config.json
register wan2.2-ti2v-5b modelfiles/wan2.2-ti2v-5b/config.json
