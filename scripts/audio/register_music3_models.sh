#!/usr/bin/env bash
# Register MiniMax Music 3 mlx lab tag (weights stay under ~/.zerollama).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
GO="${GO:-go}"
"$GO" run ./scripts/register_wan_manifest minimax-music3:lab modelfiles/minimax-music3/config.json
