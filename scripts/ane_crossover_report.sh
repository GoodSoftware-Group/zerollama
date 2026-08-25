#!/usr/bin/env bash
# Print ANE vs MPS width crossover summary for key local models (Darwin lab).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "skip: ane_crossover_report requires Darwin" >&2
  exit 0
fi

export ANE_REPO="${ANE_REPO:-${HOME}/Sites/inference/ane}"
./scripts/ane_probe_build.sh >/dev/null

go build -o "${ROOT}/.ane_crossover_bin" . 2>/dev/null
trap 'rm -f "${ROOT}/.ane_crossover_bin"' EXIT
BIN="${ROOT}/.ane_crossover_bin"

echo "== ane_crossover_report $(date -u +%Y-%m-%dT%H:%MZ) =="
echo "NOTE: MPS/Metal legs need an idle GPU. If serve or other Metal work is running, MPS times are not comparable."

summarize() {
  local label="$1"
  shift
  echo ""
  echo "### ${label}"
  "${BIN}" ane-prefill-crossover "$@" 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(f\"seq={d.get('seq')} crossover_ic={d.get('width_crossover')} ane_wins={d['ane_wins']} mps_wins={d['metal_wins']}\")
for p in d.get('points',[]):
  m=(p.get('metal_mps') or {})
  print(f\"  ic={p['ic']:4d} {p['faster']:10s} ane={p['ane']['eval_ms']:.3f} mps={m.get('eval_ms',0):.3f}\")
"
}

summarize "Global quick grid" --quick
summarize "qwen3.6:latest" --model qwen3.6 --quick
summarize "eliza-1-2b" --model eliza-1-2b --quick
summarize "tiny-agent" --model tiny-agent --quick

echo ""
echo "ane_crossover_report: done"
