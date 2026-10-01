#!/usr/bin/env bash
# Register MiniMax-H3 Wan2GP CUDA config-only manifests (weights under WAN2GP ckpts).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
GO="${GO:-go}"
"$GO" run ./scripts/register_wan_manifest minimax-h3-fl2va:lab modelfiles/minimax-h3-fl2va/config.json
"$GO" run ./scripts/register_wan_manifest minimax-h3-fl2va-pruned:lab modelfiles/minimax-h3-fl2va-pruned/config.json
echo "registered minimax-h3-fl2va:lab and minimax-h3-fl2va-pruned:lab"
