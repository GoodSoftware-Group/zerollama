#!/usr/bin/env bash
# Same-prompt compare: TRELLIS.2 (from still) vs TRELLIS text XL.
# Run on astra. Uses the free card (CUDA_VISIBLE_DEVICES default 1).
#
# Usage:
#   # after copying still.png from the Mac (same prompt as --prompt):
#   ./scripts/mesh/compare_trellis_prompt.sh \
#     --prompt "a red ceramic mug" --image still.png --out-dir ./lab-out \
#     --still-source "mlx z-image-turbo + rembg on M4"
set -euo pipefail

PROMPT=""
IMAGE=""
OUT_DIR="./trellis-lab-out"
SEED=1
RESOLUTION=1024
SKIP_TEXT=0
STILL_SOURCE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt) PROMPT="${2:-}"; shift 2 ;;
    --image) IMAGE="${2:-}"; shift 2 ;;
    --out-dir) OUT_DIR="${2:-}"; shift 2 ;;
    --seed) SEED="${2:-}"; shift 2 ;;
    --resolution) RESOLUTION="${2:-}"; shift 2 ;;
    --still-source) STILL_SOURCE="${2:-}"; shift 2 ;;
    --skip-text) SKIP_TEXT=1; shift ;;
    -h|--help)
      echo "Usage: $0 --prompt TEXT --image PNG [--out-dir DIR] [--seed N] [--resolution 1024] [--still-source DESC] [--skip-text]"
      exit 0
      ;;
    *) echo "unknown: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$PROMPT" && -n "$IMAGE" ]] || { echo "need --prompt and --image" >&2; exit 2; }
[[ -f "$IMAGE" ]] || { echo "image not found: $IMAGE" >&2; exit 2; }
if [[ -z "$STILL_SOURCE" ]]; then
  echo "error: pass --still-source describing how the PNG was made (e.g. 'mlx + rembg' or 'example T.png rembg')" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TRELLIS_ROOT="${TRELLIS_ROOT:-$HOME/.zerollama/third_party/trellis}"
# shellcheck source=/dev/null
source "$TRELLIS_ROOT/env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
STILL="$OUT_DIR/still.png"
cp -f "$IMAGE" "$STILL"
printf '%s\n' "$PROMPT" > "$OUT_DIR/prompt.txt"
printf '%s\n' "$STILL_SOURCE" > "$OUT_DIR/still-source.txt"

echo "=== TRELLIS.2 on GPU ${CUDA_VISIBLE_DEVICES} resolution=${RESOLUTION} ==="
# shellcheck source=/dev/null
source "$TRELLIS_VENV/bin/activate"
export PATH="${TRELLIS_VENV}/bin:/usr/bin:/usr/local/bin:${PATH}"
export PYTHONPATH="${TRELLIS_ROOT}/TRELLIS.2:${PYTHONPATH:-}"
python "$REPO_ROOT/scripts/mesh/trellis_mesh_generate.py" \
  --image "$STILL" \
  --out "$OUT_DIR/trellis2.glb" \
  --resolution "$RESOLUTION" \
  --seed "$SEED" \
  --decimation 100000 \
  --texture-size 1024 \
  --note-out "$OUT_DIR/trellis2.json"

python3 "$REPO_ROOT/scripts/mesh/test_glb_inspect.py" --require-pbr "$OUT_DIR/trellis2.glb"

if [[ "$SKIP_TEXT" -eq 1 ]]; then
  echo "skip text XL (--skip-text)"
  exit 0
fi

if [[ -z "${TRELLIS_TEXT_VENV:-}" || ! -x "${TRELLIS_TEXT_VENV}/bin/python" ]]; then
  echo "warning: text XL venv missing — run install_trellis_astra.sh --with-text" >&2
  echo "TRELLIS.2 half is done: $OUT_DIR/trellis2.glb"
  exit 0
fi

echo "=== TRELLIS text XL ==="
# shellcheck source=/dev/null
source "$TRELLIS_TEXT_VENV/bin/activate"
export PATH="${TRELLIS_TEXT_VENV}/bin:/usr/bin:/usr/local/bin:${PATH}"
export PYTHONPATH="${TRELLIS_TEXT_SRC}:${PYTHONPATH:-}"
python "$REPO_ROOT/scripts/mesh/trellis_text_generate.py" \
  --prompt "$PROMPT" \
  --out "$OUT_DIR/trellis-text.glb" \
  --ply-out "$OUT_DIR/trellis-text.ply" \
  --seed "$SEED" \
  --texture-size 1024

python3 - <<PY
import json
from pathlib import Path
out = Path("$OUT_DIR")
t2 = json.loads((out / "trellis2.json").read_text()) if (out / "trellis2.json").exists() else None
tx = json.loads((out / "trellis-text.json").read_text()) if (out / "trellis-text.json").exists() else None
t2_pbr = bool(t2 and (t2.get("glb_inspect") or {}).get("baseColorTexture") and (t2.get("glb_inspect") or {}).get("metallicRoughnessTexture"))
summary = {
  "prompt": Path("$OUT_DIR/prompt.txt").read_text().strip(),
  "still_source": Path("$OUT_DIR/still-source.txt").read_text().strip(),
  "seed": $SEED,
  "resolution": $RESOLUTION,
  "gpu": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
  "trellis2": t2,
  "trellis_text": tx,
  "decision": {
    "service_shape": "image_still_then_trellis2" if t2_pbr else "undecided",
    "why": (
      "trellis2.glb carries baseColorTexture + metallicRoughnessTexture (room PBR). "
      "trellis-text.glb is color-only; useful as a no-imagegen compare, not the room default."
      if t2_pbr else
      "trellis2 missing PBR textures — do not decide service shape from this run"
    ),
    "visual_review": "open both GLBs under an HDRI / environment light before shipping quality claims",
    "next": [
      "license-clean bake (replace nvdiffrast UV raster; keep cumesh UVs + PBR volume)",
      "POST /v1/meshes on astra with video_exclusive hold",
      "Mac lit model-viewer preview",
    ],
  },
}
(out / "compare-note.json").write_text(json.dumps(summary, indent=2) + "\n")
print("wrote", out / "compare-note.json")
PY

echo "Done. Inspect $OUT_DIR/trellis2.glb and $OUT_DIR/trellis-text.glb under an environment light."
