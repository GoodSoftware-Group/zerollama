#!/usr/bin/env bash
# L1 CUDA production gate — single-stream calibrate + concurrent bench + verdict.
#
# WHY: L1 exit needs both legs. Single-stream can be flat (+0.7% on 9B); concurrent
# N=2 is where n_parallel=2 pays off (+10.5% on eliza-1 9B @ CT 1564 Jun 2026).
#
# Usage:
#   CUDA_LLAMA_MODEL=/root/eliza-1-9b-256k.gguf ./scripts/l1_cuda_full_gate.sh
#   CUDA_LLAMA_MODEL=/path/to/supernova-fp16.gguf ./scripts/l1_cuda_full_gate.sh  # optional re-validate
#
# Env:
#   CUDA_LLAMA_MODEL / LLAMA_MODEL  — production GGUF (required)
#   L1_GATE_DIR                     — artifact root (default /tmp/l1-production-gate)
#   L1_GATE_OUT                     — merged JSON report (default ${L1_GATE_DIR}/gate.json)
#   L1C_N                           — concurrent count (default 2)
#   L1_SKIP_CALIBRATE=1             — concurrent-only re-run
#   L1_SKIP_CONCURRENT=1            — calibrate-only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "warn: l1_cuda_full_gate targets Linux CUDA; continuing anyway" >&2
fi

LLAMA_MODEL="${CUDA_LLAMA_MODEL:-${LLAMA_MODEL:-}}"
if [[ -z "${LLAMA_MODEL:-}" ]]; then
  echo "Set CUDA_LLAMA_MODEL to a production GGUF (7B–9B class on 16GB; eliza-1 9B is ship proxy)" >&2
  exit 1
fi
export CUDA_LLAMA_MODEL="${LLAMA_MODEL}"

L1_GATE_DIR="${L1_GATE_DIR:-/tmp/l1-production-gate}"
L1_GATE_OUT="${L1_GATE_OUT:-${L1_GATE_DIR}/gate.json}"
mkdir -p "${L1_GATE_DIR}"

echo "== L1 CUDA full gate =="
echo "model: ${LLAMA_MODEL}"
echo "artifacts: ${L1_GATE_DIR}/"
echo "report: ${L1_GATE_OUT}"
echo ""

if [[ "${L1_SKIP_CALIBRATE:-0}" != "1" ]]; then
  L1_OUT_DIR="${L1_GATE_DIR}/calibrate" \
    "${ROOT}/scripts/l1_cuda_calibrate.sh"
fi

if [[ "${L1_SKIP_CONCURRENT:-0}" != "1" ]]; then
  L1C_OUT_DIR="${L1_GATE_DIR}/concurrent" \
  L1C_ENFORCE=0 \
    "${ROOT}/scripts/l1_cuda_concurrent_bench.sh"
fi

python3 <<PY
import json
from pathlib import Path

gate_dir = Path("${L1_GATE_DIR}")
cal_dir = gate_dir / "calibrate"
con_dir = gate_dir / "concurrent"

def load_calibrate():
    rows = {}
    if not cal_dir.is_dir():
        return rows
    for p in sorted(cal_dir.glob("*.json")):
        d = json.loads(p.read_text())
        leg = d["legs"]["stock"]
        b = leg["bench"]
        gp = leg.get("gpu_profile") or {}
        rows[p.stem] = {
            "tok_s": b.get("decode_tok_per_s"),
            "n_parallel": gp.get("n_parallel"),
            "gpu_profile_id": gp.get("id"),
        }
    return rows

def load_concurrent():
    rows = {}
    if not con_dir.is_dir():
        return rows
    for p in sorted(con_dir.glob("*.json")):
        d = json.loads(p.read_text())
        gp = d.get("gpu_profile") or {}
        rows[d.get("label", p.stem)] = {
            "agg_tok_s": d.get("agg_tok_s_mean"),
            "n_concurrent": d.get("n_concurrent"),
            "n_parallel": gp.get("n_parallel"),
            "gpu_profile_id": gp.get("id"),
        }
    return rows

report = {
    "model": "${LLAMA_MODEL}",
    "single_stream_min_delta_pct": 0.0,
    "legs": {
        "calibrate": ${L1_SKIP_CALIBRATE:-0} != 1,
        "concurrent": ${L1_SKIP_CONCURRENT:-0} != 1,
    },
    "calibrate": load_calibrate(),
    "concurrent": load_concurrent(),
}
Path("${L1_GATE_OUT}").write_text(json.dumps(report, indent=2) + "\n")
print(f"wrote ${L1_GATE_OUT}")
PY

"${ROOT}/scripts/l1_gate_report.sh" "${L1_GATE_OUT}"
