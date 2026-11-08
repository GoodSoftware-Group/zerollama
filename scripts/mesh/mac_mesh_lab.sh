#!/usr/bin/env bash
# Lab-resident TRELLIS.2 on Metal: load once, bake many stills.
#
# Not a zerollama sidecar. Default bind 127.0.0.1:18091 (never 11434 / 8081).
#
#   ./scripts/mesh/mac_mesh_lab.sh
#   printf '%s\n' '{"image":"/tmp/trellis-lab/still_mug_photo.png","out":".../a.glb"}' \
#     | nc 127.0.0.1 18091
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TRELLIS_ROOT="${TRELLIS_ROOT:-$HOME/.zerollama/third_party/trellis-mac}"
HOST="${TRELLIS_LAB_HOST:-127.0.0.1}"
PORT="${TRELLIS_LAB_PORT:-18091}"

if [[ "$PORT" == "11434" || "$PORT" == "8081" ]]; then
  echo "error: refused production port $PORT (use 18091)" >&2
  exit 2
fi

if [[ -f "$TRELLIS_ROOT/env-mac.sh" ]]; then
  # shellcheck disable=SC1091
  source "$TRELLIS_ROOT/env-mac.sh"
fi

export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
export SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-sdpa}"

exec python "$REPO_ROOT/scripts/mesh/trellis_mesh_lab_mac.py" \
  --host "$HOST" \
  --port "$PORT" \
  "$@"
