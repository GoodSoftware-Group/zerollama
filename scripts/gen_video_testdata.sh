#!/usr/bin/env bash
# Generate a tiny H.264 MP4 for optional local ffmpeg golden runs (Phase D).
#
# WHY: Checked-in video blobs bloat git; operators/CI with ffmpeg can materialize
# the same lavfi fixture the unit test uses when debugging sampling parity.
#
# Usage:
#   ./scripts/gen_video_testdata.sh
#   go test ./server/modality/... -run ffmpegGolden -count=1
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT}/server/modality/testdata"
OUT="${OUT_DIR}/lavfi_1s_64x64.mp4"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i "testsrc=duration=1:size=64x64:rate=10" \
  -pix_fmt yuv420p "${OUT}"

echo "wrote ${OUT}"
echo "optional: VIDEO_TESTDATA=${OUT} go test ./server/modality/... -run ffmpegGolden -v"
