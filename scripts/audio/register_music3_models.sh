#!/usr/bin/env bash
# Register minimax-music3:lab.
# Darwin: mlx-audio pack (modelfiles/minimax-music3).
# Linux: Wan2GP CUDA pack (modelfiles/minimax-music3-wan2gp). Weights stay on disk.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
GO="${GO:-go}"
if [[ "$(uname -s)" == "Darwin" ]]; then
  CFG=modelfiles/minimax-music3/config.json
else
  CFG=modelfiles/minimax-music3-wan2gp/config.json
fi
"$GO" run ./scripts/register_wan_manifest minimax-music3:lab "$CFG"
