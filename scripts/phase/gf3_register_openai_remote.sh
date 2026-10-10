#!/usr/bin/env bash
# GF3 — register config-only glm-5.3-flash tag → openai-remote sidecar.
# Does not start serve. Lab smoke uses a non-production OLLAMA_HOST.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NAME="${1:-glm-5.3-flash}"
CFG="${2:-$ROOT/scripts/phase/gf3_openai_remote_config.json}"
cd "$ROOT"
go run ./scripts/register_wan_manifest "$NAME" "$CFG"
echo "registered $NAME (openai-remote → see $CFG)"
