#!/usr/bin/env bash
# Video expansion + session cache smoke (native path, no GPU).
#
# WHY: SGLang xfer Tier 1 added URL LRU, global expand LRU, and session expand LRU.
# This script proves cache + preflight + span wiring in CI/operator shells without
# a full VLM model or ffmpeg fixture binaries in git.
#
# Usage:
#   ./scripts/video_expand_cache_smoke.sh
#
# Optional live server (manual):
#   ./zerollama serve &
#   # send chat with same video_url + options.prompt_cache_key twice; grep logs for:
#   #   video url fetch cache hit
#   #   video sample session cache hit
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "== video expand cache unit gate =="
go test ./server/modality/... -count=1 -run 'Video|Session|Expand|Preflight|Policy|FFmpeg|Agent|Preprocessed|Padded' -short

echo "== runner ViT session embed overlay =="
go test ./runner/llamarunner/... -count=1 -run 'TestSessionEmbed|TestGrowCache' -short

echo "== openai video fetch cache =="
go test ./openai/... -count=1 -run 'VideoURL' -short

echo "PASS video_expand_cache_smoke"
