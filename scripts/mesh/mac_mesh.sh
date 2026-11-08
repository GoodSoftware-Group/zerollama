#!/usr/bin/env bash
# Optional Mac chain: still (if serve up) → TRELLIS.2 MPS → PBR inspect.
#
# Finish line does NOT require this script. Prefer an existing RGBA still:
#   source ~/.zerollama/third_party/trellis-mac/env-mac.sh
#   python scripts/mesh/trellis_mesh_generate_mac.py --image … --out …
#
# Usage:
#   ./scripts/mesh/mac_mesh.sh --image /tmp/trellis-lab/still_mug.png
#   ./scripts/mesh/mac_mesh.sh --prompt "a red ceramic mug"   # needs serve at OLLAMA_HOST
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TRELLIS_ROOT="${TRELLIS_ROOT:-$HOME/.zerollama/third_party/trellis-mac}"
OUT_DIR="${OUT_DIR:-$TRELLIS_ROOT/lab/out}"
OUT_GLB="$OUT_DIR/trellis2-mps.glb"
STILL=""
PROMPT=""
RESOLUTION=512
SEED=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) STILL="${2:-}"; shift 2 ;;
    --prompt) PROMPT="${2:-}"; shift 2 ;;
    --out-dir) OUT_DIR="${2:-}"; OUT_GLB="$OUT_DIR/trellis2-mps.glb"; shift 2 ;;
    --resolution) RESOLUTION="${2:-}"; shift 2 ;;
    --seed) SEED="${2:-}"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 (--image PATH | --prompt TEXT) [--out-dir DIR] [--resolution 512|1024]"
      exit 0
      ;;
    *) echo "unknown: $1" >&2; exit 1 ;;
  esac
done

if [[ -f "$TRELLIS_ROOT/env-mac.sh" ]]; then
  # shellcheck disable=SC1091
  source "$TRELLIS_ROOT/env-mac.sh"
fi

mkdir -p "$OUT_DIR"

if [[ -z "$STILL" ]]; then
  if [[ -z "$PROMPT" ]]; then
    if [[ -f /tmp/trellis-lab/still_mug_connected.png ]]; then
      STILL=/tmp/trellis-lab/still_mug_connected.png
    elif [[ -f /tmp/trellis-lab/still_mug.png ]]; then
      echo "warning: using disconnected still_mug.png — prefer make_connected_still.py" >&2
      STILL=/tmp/trellis-lab/still_mug.png
    else
      echo "error: pass --image or --prompt (or run make_connected_still.py)" >&2
      exit 2
    fi
  else
    STILL="$OUT_DIR/still.png"
    "$REPO_ROOT/scripts/mesh/mac_object_still.sh" --prompt "$PROMPT" --out "$STILL"
  fi
fi

python "$REPO_ROOT/scripts/mesh/trellis_mesh_generate_mac.py" \
  --image "$STILL" \
  --out "$OUT_GLB" \
  --resolution "$RESOLUTION" \
  --seed "$SEED" \
  --note-out "$OUT_DIR/trellis2-mps.json"

python3 "$REPO_ROOT/scripts/mesh/test_glb_inspect.py" --require-pbr --require-png "$OUT_GLB"
echo "OK: $OUT_GLB"
