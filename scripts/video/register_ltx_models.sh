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
register ltxv-2b-distilled:lab modelfiles/ltxv-2b-distilled/config.json
register ltxv-2b-mlx:lab modelfiles/ltxv-2b-mlx/config.json
register ltxv-13b-mlx:lab modelfiles/ltxv-13b-mlx/config.json
# LTX-2.5 distilled (Gemma4 TE + start/end keyframes / control). Needs ~48 GiB host + 2×4090 class.
register ltx2.5-22b-distilled:48g modelfiles/ltx2.5-22b-distilled/config.json
