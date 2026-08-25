#!/usr/bin/env bash
# Register MiniMax-H3 T2VA config-only manifests (weights stay under ~/.zerollama).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
GO="${GO:-go}"
"$GO" run ./scripts/register_wan_manifest minimax-h3-tiny:lab modelfiles/minimax-h3-tiny/config.json
"$GO" run ./scripts/register_wan_manifest minimax-h3-768:lab modelfiles/minimax-h3-768/config.json
