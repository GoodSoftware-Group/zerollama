#!/usr/bin/env bash
# Dry-run + optional short GPU smoke for LTX-2.5 distilled (Wan2GP).
# Usage:
#   ./scripts/video/smoke_ltx2_wan2gp.sh              # dry-run only
#   RUN_GPU=1 ./scripts/video/smoke_ltx2_wan2gp.sh    # short generate (needs free GPU)
# Never binds :11434 / :8081.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${WAN2GP_ROOT:-}" ]]; then
  if [[ -d /mnt/ssd2/zerollama/third_party/wan2gp ]]; then
    WAN2GP_ROOT=/mnt/ssd2/zerollama/third_party/wan2gp
  else
    WAN2GP_ROOT="$HOME/.zerollama/third_party/wan2gp"
  fi
fi
WAN2GP_REPO="${WAN2GP_REPO:-$WAN2GP_ROOT/repo}"
WAN2GP_CKPT_DIR="${WAN2GP_CKPT_DIR:-$WAN2GP_ROOT/ckpts}"
WAN2GP_VENV="${WAN2GP_VENV:-$WAN2GP_ROOT/venv}"
PY="${WAN2GP_VENV}/bin/python3"
OUT="${LTX_SMOKE_OUT:-/mnt/ssd2/zerollama/ltx25_smoke.mp4}"
if [[ ! -d "$(dirname "$OUT")" ]]; then
  OUT=/tmp/ltx25_smoke.mp4
fi

if [[ ! -x "$PY" ]]; then
  echo "missing venv python at $PY — run ./scripts/video/install_ltx2_wan2gp.sh --venv-only" >&2
  exit 1
fi

echo "==> dry-run"
WAN2GP_REPO="$WAN2GP_REPO" WAN2GP_CKPT_DIR="$WAN2GP_CKPT_DIR" \
  LTX_PROMPT='dry-run probe: a red ball on a wooden table, soft daylight' \
  LTX_SIZE=1280x704 LTX_FRAMES=97 LTX_STEPS=8 \
  LTX_MODEL_TYPE=ltx2_25_22B_distilled \
  LTX_OUTPUT_PATH=/tmp/ltx2-dry.mp4 LTX_DRY_RUN=1 \
  "$PY" scripts/video/ltx_video_generate.py

if [[ "${RUN_GPU:-0}" != "1" ]]; then
  echo "dry-run OK (set RUN_GPU=1 for short generate → $OUT)"
  exit 0
fi

FRAMES="${LTX_SMOKE_FRAMES:-17}"
SIZE="${LTX_SMOKE_SIZE:-768x512}"
STEPS="${LTX_SMOKE_STEPS:-8}"
echo "==> GPU smoke frames=$FRAMES size=$SIZE steps=$STEPS → $OUT"
WAN2GP_REPO="$WAN2GP_REPO" WAN2GP_CKPT_DIR="$WAN2GP_CKPT_DIR" \
  LTX_PROMPT='A ceramic mug on a sunlit kitchen counter. Camera slowly pushes in. Soft natural light, shallow depth of field.' \
  LTX_SIZE="$SIZE" LTX_FRAMES="$FRAMES" LTX_STEPS="$STEPS" \
  LTX_MODEL_TYPE=ltx2_25_22B_distilled LTX_FPS=24 \
  LTX_OUTPUT_PATH="$OUT" LTX_MMGP_PROFILE="${LTX_MMGP_PROFILE:-5}" \
  "$PY" scripts/video/ltx_video_generate.py

ls -lh "$OUT"
echo "OK $OUT"
