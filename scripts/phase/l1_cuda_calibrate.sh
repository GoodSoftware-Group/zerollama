#!/usr/bin/env bash
# L1 CUDA calibration — profile OFF baseline vs ON (+ optional n_parallel sweep).
#
# WHY: eliza-ported rtx-5080.json may regress single-stream tok/s (e.g. -np 4 on 1B).
# Run on a production-sized GGUF (7B–14B Q4/Q8) before editing runtime/configs/gpu/rtx-5080.json.
#
# Usage:
#   CUDA_LLAMA_MODEL=/root/eliza-1-9b-256k.gguf ./scripts/phase/l1_cuda_calibrate.sh
#   L1_SWEEP_NP=1,2,4 ./scripts/phase/l1_cuda_calibrate.sh   # extra ON legs with -np override
#
# Env:
#   CUDA_LLAMA_MODEL     — GGUF path (required)
#   L1_OUT_DIR           — default /tmp/l1-cuda-calibrate
#   L1_NUM_CTX           — default 8192
#   L1_NUM_PREDICT       — default 128
#   L1_BENCH_RUNS        — default 2
#   L1_SWEEP_NP          — comma list for ON legs via LLAMA_SERVER_EXTRA_ARGS=-np N
#   L1_SKIP_OFF=1        — skip profile-off baseline
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/runtime/linux_runtime_serve_lib.sh
source "${ROOT}/scripts/runtime/linux_runtime_serve_lib.sh"

# WHY: embedded zerollama on :8081 races the uv sidecar and poisons concurrent bench.
linux_runtime_stop_sidecar_port
fuser -k 8080/tcp 2>/dev/null || true
for _zpid in $(pgrep -x zerollama 2>/dev/null || true); do kill -9 "$_zpid" 2>/dev/null || true; done
sleep 1

OUT_DIR="${L1_OUT_DIR:-/tmp/l1-cuda-calibrate}"
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

if [[ -z "${CUDA_LLAMA_MODEL:-}" && -z "${LLAMA_MODEL:-}" ]]; then
  echo "Set CUDA_LLAMA_MODEL to a production GGUF (e.g. 7B–9B Q8 on 16GB)" >&2
  exit 1
fi
export CUDA_LLAMA_MODEL="${CUDA_LLAMA_MODEL:-${LLAMA_MODEL}}"
export L2_NUM_CTX="${L1_NUM_CTX:-8192}"
export L2_NUM_PREDICT="${L1_NUM_PREDICT:-128}"
export L2_BENCH_RUNS="${L1_BENCH_RUNS:-2}"
export L2_SKIP_FORK=1
export L2_SKIP_PREFILL=1
# WHY stock cache only: L1 tunes batch/np/-fa — fork QJL is L2 (see docs/gpu-profiles-l1.md).
export ZEROLLAMA_LLAMA_FORK=0
# WHY no profile -c: rtx-5080 default -c 32768 pre-allocates KV and regresses 8k bench vs OFF np=1.
export ZEROLLAMA_GPU_PROFILE_CTX=0
l1_export_llama_binary_env "${ROOT}"

_run_leg() {
  local label="$1"
  local profile="$2"   # 0 | 1
  local extra_args="${3:-}"
  local out="${OUT_DIR}/${label}.json"

  echo ""
  echo "== L1 cal: ${label} (profile=${profile} extra='${extra_args}') =="

  env \
    ZEROLLAMA_GPU_PROFILE="${profile}" \
    ZEROLLAMA_LLAMA_FORK=0 \
    ZEROLLAMA_GPU_PROFILE_CTX=0 \
    LLAMA_SERVER_EXTRA_ARGS="${extra_args}" \
    L2_CUDA_BENCH_OUT="${out}" \
    L2_STOCK_FORK_MODE=off \
    CUDA_LLAMA_MODEL="${CUDA_LLAMA_MODEL}" \
    "${ROOT}/scripts/phase/l2_cuda_bench.sh" 2>&1 | tail -8

  python3 -c "
import json, sys
d = json.load(open('${out}'))
leg = d['legs']['stock']
gp = leg.get('gpu_profile') or {}
b = leg['bench']
print(f\"  decode: {b['decode_tok_per_s']} tok/s  gpu_profile={gp.get('id')} n_parallel={gp.get('n_parallel')}\")
print(f\"  llama_args: {' '.join(leg.get('llama_server_args') or [])}\")
"
}

echo "== L1 CUDA calibration =="
echo "model: ${CUDA_LLAMA_MODEL}"
echo "ctx=${L2_NUM_CTX} predict=${L2_NUM_PREDICT} runs=${L2_BENCH_RUNS}"
echo "out: ${OUT_DIR}/"

if [[ "${L1_SKIP_OFF:-0}" != "1" ]]; then
  _run_leg "profile-off" "0" ""
fi

_run_leg "profile-on-default" "1" ""

IFS=',' read -r -a _np_sweep <<< "${L1_SWEEP_NP:-}"
for np in "${_np_sweep[@]}"; do
  np="${np// /}"
  [[ -z "${np}" ]] && continue
  _run_leg "profile-on-np${np}" "1" "-np ${np}"
done

echo ""
echo "== L1 calibration summary =="
python3 <<PY
import json
import os
from pathlib import Path

out_dir = Path("${OUT_DIR}")
rows = []
for p in sorted(out_dir.glob("*.json")):
    d = json.loads(p.read_text())
    leg = d["legs"]["stock"]
    gp = leg.get("gpu_profile") or {}
    b = leg["bench"]
    rows.append({
        "label": p.stem,
        "tok_s": b["decode_tok_per_s"],
        "n_parallel": gp.get("n_parallel"),
        "gpu_profile_id": gp.get("id"),
    })

off = next((r for r in rows if r["label"] == "profile-off"), None)
print(f"{'label':<22} {'tok/s':>8}  {'n_parallel':>10}  vs OFF")
for r in rows:
    delta = ""
    if off and off["tok_s"] and r["tok_s"]:
        pct = (r["tok_s"] - off["tok_s"]) / off["tok_s"] * 100
        delta = f"{pct:+.1f}%"
    print(f"{r['label']:<22} {r['tok_s']:>8.2f}  {str(r['n_parallel'] or '-'):>10}  {delta}")

best = max(rows, key=lambda r: r["tok_s"] or 0)
print()
print(f"best: {best['label']} @ {best['tok_s']} tok/s")
if off:
    winners = [r for r in rows if r["tok_s"] and off["tok_s"] and r["tok_s"] >= off["tok_s"]]
    if winners:
        best_on = max((r for r in winners if r["label"] != "profile-off"), key=lambda r: r["tok_s"], default=None)
        if best_on:
            print(f"VERDICT: profile ON can match/beat OFF — best ON leg: {best_on['label']} @ {best_on['tok_s']} tok/s")
        else:
            print("VERDICT: only OFF wins — tune rtx-5080.json batch/np")
    else:
        print("VERDICT: profile still loses vs OFF — tune batch/np in rtx-5080.json")
PY

echo ""
echo "Next: edit runtime/configs/gpu/rtx-5080.json from best leg; rerun this script."
echo "Doc: docs/gpu-profiles-l1.md"
