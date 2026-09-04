#!/usr/bin/env bash
# ltx_post_anime.sh — look presets for LTX-MLX anime renders.
#
# Looks:
#   LTX_LOOK=90s   (default) 1990s cel-anime grade: soft lines, warm analog
#                  curves, vignette, animated film grain. No posterization —
#                  90s anime has rich color, just graded + grain.
#   LTX_LOOK=eva   Evangelion-era TV-cel: stronger line restore, no vignette.
#   LTX_LOOK=cel   limited-palette poster look: temporal denoise → line
#                  sharpen → grade → k-color palette quantization.
#   LTX_LOOK=off   pass-through (re-encode only).
#
# Usage:
#   ./scripts/video/ltx_post_anime.sh IN.mp4 OUT.mp4 [colors]   # colors = cel only
# Env:
#   LTX_LOOK=90s|eva|cel|off
#   LTX_PALETTE=path.png  (cel) reuse a fixed palette across shots
#   LTX_COLORS=N          (cel) default 32
#   LTX_DITHER=bayer      (cel) bayer|floyd_steinberg|none
#   LTX_GRAIN=N           (90s) noise strength, default 5
set -euo pipefail

IN="${1:?input mp4}"
OUT="${2:?output mp4}"
LOOK="${LTX_LOOK:-90s}"
COLORS="${3:-${LTX_COLORS:-32}}"
DITHER="${LTX_DITHER:-bayer}"
PALETTE_FILE="${LTX_PALETTE:-}"
GRAIN="${LTX_GRAIN:-5}"

FF="$(command -v ffmpeg || echo ~/.homebrew/bin/ffmpeg)"

case "$LOOK" in
90s)
  # Why: 90s cel anime = soft line weight (not bold poster outlines), warm
  # analog color (red-lifted mids, blue-rolled shadows), vignette, animated
  # grain. Light denoise first so grain lands on clean plates.
  "$FF" -v error -i "$IN" -vf \
    "hqdn3d=1:1:4:4,unsharp=3:3:0.35,curves=r='0/0.03 0.5/0.56 1/1':b='0/0.06 0.5/0.47 1/0.98',eq=saturation=1.02,vignette=PI/4.5,noise=alls=${GRAIN}:allf=t+u" \
    -c:v libx264 -crf 18 -y "$OUT"
  echo "ltx_post_anime: 90s grade -> $OUT"
  ;;
eva)
  # Hard-shadow TV-cel interiors. Stronger unsharp, no vignette/warm curves.
  "$FF" -v error -i "$IN" -vf \
    "hqdn3d=1:1:3:3,unsharp=5:5:0.7,eq=saturation=1.1:contrast=1.04,noise=alls=3:allf=t+u" \
    -c:v libx264 -crf 18 -y "$OUT"
  echo "ltx_post_anime: eva cel -> $OUT"
  ;;
cel)
  graded=$(mktemp /tmp/ltx_graded_XXXXXX.mp4)
  trap 'rm -f "$graded"' EXIT
  # Temporal denoise stabilizes frames for quantization; unsharp restores
  # line weight; eq pushes poster contrast/saturation.
  "$FF" -v error -i "$IN" \
    -vf "hqdn3d=1.5:1.5:6:6,unsharp=5:5:0.8,eq=contrast=1.08:saturation=1.15" \
    -c:v libx264 -crf 18 -y "$graded"
  if [[ -n "$PALETTE_FILE" && -f "$PALETTE_FILE" ]]; then
    PAL="$PALETTE_FILE"
  else
    PAL=$(mktemp /tmp/ltx_pal_XXXXXX.png)
    # stats_mode=diff weights the palette toward high-motion (subject) frames.
    "$FF" -v error -i "$graded" \
      -vf "palettegen=max_colors=${COLORS}:stats_mode=diff" -y "$PAL"
  fi
  "$FF" -v error -i "$graded" -i "$PAL" \
    -lavfi "paletteuse=dither=${DITHER}:bayer_scale=2" \
    -c:v libx264 -crf 18 -y "$OUT"
  [[ -n "${PALETTE_FILE:-}" ]] || rm -f "$PAL"
  echo "ltx_post_anime: ${COLORS}-color cel -> $OUT"
  ;;
off)
  "$FF" -v error -i "$IN" -c:v libx264 -crf 18 -y "$OUT"
  echo "ltx_post_anime: pass-through -> $OUT"
  ;;
*)
  echo "unknown LTX_LOOK=$LOOK (90s|eva|cel|off)" >&2
  exit 1
  ;;
esac
