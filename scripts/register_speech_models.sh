#!/usr/bin/env bash
# Register Piper / Whisper / remote-tts (Chatterbox, Orpheus, Kokoro) manifests.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
GO="${GO:-go}"
export GOFLAGS="${GOFLAGS:--mod=mod}"

"$GO" run ./scripts/register_wan_manifest piper-lessac:latest modelfiles/piper-lessac/config.json
"$GO" run ./scripts/register_wan_manifest whisper-base:latest modelfiles/whisper-base/config.json
"$GO" run ./scripts/register_wan_manifest chatterbox:latest modelfiles/chatterbox/config.json
"$GO" run ./scripts/register_wan_manifest orpheus:latest modelfiles/orpheus/config.json
"$GO" run ./scripts/register_wan_manifest kokoro:latest modelfiles/kokoro/config.json

echo "Registered speech tags: piper-lessac, whisper-base, chatterbox, orpheus, kokoro"
echo "List voices: curl -s localhost:11434/v1/audio/voices | jq ."
