#!/usr/bin/env bash
# T8 training loss-curve fixture (CPU). Compares padding_free vs longest at
# batch_size=1 and dumps packing / multi-batch curves.
#
# Usage (repo root):
#   ./scripts/training/loss_curve_fixture.sh
#   STEPS=8 ./scripts/training/loss_curve_fixture.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -x "${ROOT}/.venv-training/bin/python" ]]; then
  PY="${ROOT}/.venv-training/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "No python found" >&2
  exit 1
fi

STEPS="${STEPS:-4}"
SEED="${SEED:-0}"
OUT="${OUT:-${ROOT}/tests/fixtures/.sft_loss_curves.json}"

exec "$PY" "${ROOT}/training_loss_fixture.py" --steps "$STEPS" --seed "$SEED" --out "$OUT"
