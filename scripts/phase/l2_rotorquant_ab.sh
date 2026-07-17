#!/usr/bin/env bash
# L2 RotorQuant / planar-iso A/B — multi-leg KV cache decode + VRAM on lab ports.
#
# WHY: scrya/ParaMind claim planar3/iso3 beat TurboQuant on PPL + prefill/decode.
# Measure against our unified pin profiles (q8_0, tbq, qjl/polar) before cherry-picking.
#
# Requires a RotorQuant-capable llama-server for planar*/iso* legs (see docs/llama-fork-watchlist.md).
# Our vendor binary covers stock / TBQ / QJL legs.
#
# Lab only — never binds :11434 or :8081.
#
#   # Build RotorQuant fork once (sibling checkout):
#   git clone -b feature/planarquant-kv-cache \
#     https://github.com/johndpope/llama-cpp-turboquant.git ../llama-cpp-rotorquant
#   cmake -S ../llama-cpp-rotorquant -B ../llama-cpp-rotorquant/build \
#     -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build ../llama-cpp-rotorquant/build -j --target llama-server
#
#   CUDA_LLAMA_MODEL=/path/to.gguf \
#   ROTORQUANT_LLAMA_SERVER_BIN=../llama-cpp-rotorquant/build/bin/llama-server \
#     ./scripts/phase/l2_rotorquant_ab.sh
#
# Env:
#   CUDA_LLAMA_MODEL / LLAMA_MODEL — GGUF (required)
#   LLAMA_SERVER_BIN / LLAMA_CPP_ROOT — our unified binary (stock/tbq/qjl legs)
#   ROTORQUANT_LLAMA_SERVER_BIN — binary with planar3/iso3 (required unless L2_RQ_LEGS omits them)
#   L2_RQ_LEGS — comma list (default: stock,tbq,qjl,planar3,iso3)
#   L2_NUM_CTX / L2_NUM_PREDICT / L2_BENCH_RUNS / L2_HIGH_CTX_WARMUPS
#   L2_PORT — default 18082 (lab)
#   CUDA_VISIBLE_DEVICES — default 1 on multi-GPU
#   L2_RQ_OUT — summary JSON (default /tmp/l2-rotorquant-ab.json)
#   L2_RQ_ALSO_LLAMA_BENCH=1 — if llama-bench sits next to the active binary, run -p/-n prefill too
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/runtime/linux_runtime_serve_lib.sh
source "${ROOT}/scripts/runtime/linux_runtime_serve_lib.sh"

LLAMA_MODEL="${CUDA_LLAMA_MODEL:-${LLAMA_MODEL:-}}"
if [[ -z "${LLAMA_MODEL}" || ! -f "${LLAMA_MODEL}" ]]; then
  echo "Set CUDA_LLAMA_MODEL to a GGUF path" >&2
  exit 1
fi

_resolve_unified_bin() {
  if [[ -n "${LLAMA_SERVER_BIN:-}" && -x "${LLAMA_SERVER_BIN}" ]]; then
    echo "${LLAMA_SERVER_BIN}"
    return 0
  fi
  if [[ -n "${LLAMA_CPP_ROOT:-}" && -x "${LLAMA_CPP_ROOT}/build/bin/llama-server" ]]; then
    echo "${LLAMA_CPP_ROOT}/build/bin/llama-server"
    return 0
  fi
  local vr
  if vr="$(l1_vendor_llama_cpp_root "${ROOT}" 2>/dev/null)" && [[ -x "${vr}/build/bin/llama-server" ]]; then
    echo "${vr}/build/bin/llama-server"
    return 0
  fi
  if [[ -x /usr/local/lib/ollama/llama-server ]]; then
    echo /usr/local/lib/ollama/llama-server
    return 0
  fi
  return 1
}

if ! UNIFIED_BIN="$(_resolve_unified_bin)"; then
  echo "Set LLAMA_SERVER_BIN or LLAMA_CPP_ROOT (built unified llama-server)" >&2
  exit 1
fi

ROTOR_BIN="${ROTORQUANT_LLAMA_SERVER_BIN:-}"
L2_RQ_LEGS="${L2_RQ_LEGS:-stock,tbq,qjl,planar3,iso3}"
L2_NUM_CTX="${L2_NUM_CTX:-8192}"
L2_NUM_PREDICT="${L2_NUM_PREDICT:-64}"
L2_BENCH_RUNS="${L2_BENCH_RUNS:-2}"
L2_HIGH_CTX_WARMUPS="${L2_HIGH_CTX_WARMUPS:-2}"
L2_PORT="${L2_PORT:-18082}"
L2_OUT="${L2_RQ_OUT:-/tmp/l2-rotorquant-ab.json}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

case "${L2_PORT}" in
  11434|8081)
    echo "Refusing production port ${L2_PORT}; use a lab port (default 18082)" >&2
    exit 1
    ;;
esac

_LEG_DIR="$(mktemp -d /tmp/l2-rotorquant-ab-XXXXXX)"
_SERVER_PID=""

cleanup() {
  [[ -n "${_SERVER_PID}" ]] && kill "${_SERVER_PID}" 2>/dev/null || true
  fuser -k "${L2_PORT}/tcp" 2>/dev/null || true
  rm -rf "${_LEG_DIR}"
}
trap cleanup EXIT

_leg_cache() {
  # stdout: bin|cache_k|cache_v
  local leg="$1"
  case "${leg}" in
    stock)   echo "${UNIFIED_BIN}|q8_0|q8_0" ;;
    tbq)     echo "${UNIFIED_BIN}|tbq4_0|tbq3_0" ;;
    qjl)     echo "${UNIFIED_BIN}|qjl1_256|q4_polar" ;;
    planar3|iso3|planar4|iso4)
      if [[ -z "${ROTOR_BIN}" || ! -x "${ROTOR_BIN}" ]]; then
        echo "Leg ${leg} needs ROTORQUANT_LLAMA_SERVER_BIN (executable)" >&2
        return 1
      fi
      echo "${ROTOR_BIN}|${leg}|${leg}"
      ;;
    planar3_f16)
      if [[ -z "${ROTOR_BIN}" || ! -x "${ROTOR_BIN}" ]]; then
        echo "Leg ${leg} needs ROTORQUANT_LLAMA_SERVER_BIN" >&2
        return 1
      fi
      echo "${ROTOR_BIN}|planar3|f16"
      ;;
    *)
      echo "Unknown L2_RQ_LEGS entry: ${leg}" >&2
      return 1
      ;;
  esac
}

_start_server() {
  local bin="$1" cache_k="$2" cache_v="$3" log="$4"
  fuser -k "${L2_PORT}/tcp" 2>/dev/null || true
  sleep 1
  export LLAMA_SERVER_BIN="${bin}"
  export LLAMA_CPP_ROOT="$(cd "$(dirname "${bin}")/../.." && pwd)"
  linux_runtime_export_llama_ld_path 2>/dev/null || true
  echo "starting ${bin} -c ${L2_NUM_CTX} cache=${cache_k}/${cache_v} port=${L2_PORT}"
  "${bin}" -m "${LLAMA_MODEL}" -c "${L2_NUM_CTX}" -ngl 99 -fa on \
    --cache-type-k "${cache_k}" --cache-type-v "${cache_v}" \
    --host 127.0.0.1 --port "${L2_PORT}" -np 1 -b 2048 -ub 512 \
    >"${log}" 2>&1 &
  _SERVER_PID=$!
  local i
  for i in $(seq 1 90); do
    if curl -sf -m 2 "http://127.0.0.1:${L2_PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "${_SERVER_PID}" 2>/dev/null; then
      echo "llama-server exited; log:" >&2
      tail -50 "${log}" >&2
      return 1
    fi
    sleep 2
  done
  echo "llama-server health timeout; log:" >&2
  tail -50 "${log}" >&2
  return 1
}

_optional_llama_bench() {
  local bin="$1" cache_k="$2" cache_v="$3" outf="$4"
  [[ "${L2_RQ_ALSO_LLAMA_BENCH:-0}" == "1" ]] || return 0
  local bench
  bench="$(dirname "${bin}")/llama-bench"
  [[ -x "${bench}" ]] || return 0
  echo "  llama-bench -ctk ${cache_k} -ctv ${cache_v} ..."
  "${bench}" -m "${LLAMA_MODEL}" -ngl 99 -fa 1 \
    -ctk "${cache_k}" -ctv "${cache_v}" \
    -p 512 -n 128 -r 2 \
    >"${outf}" 2>&1 || true
}

_bench_leg() {
  local label="$1"
  local spec bin cache_k cache_v
  spec="$(_leg_cache "${label}")" || return 1
  IFS='|' read -r bin cache_k cache_v <<<"${spec}"
  local log="${_LEG_DIR}/${label}.log"
  local out_json="${_LEG_DIR}/${label}.json"
  local bench_log="${_LEG_DIR}/${label}.bench.txt"
  echo ""
  echo "== L2 RotorQuant leg: ${label} bin=$(basename "${bin}") cache=${cache_k}/${cache_v} =="
  _start_server "${bin}" "${cache_k}" "${cache_v}" "${log}"
  L2_PORT="${L2_PORT}" L2_NUM_PREDICT="${L2_NUM_PREDICT}" L2_BENCH_RUNS="${L2_BENCH_RUNS}" \
    L2_HIGH_CTX_WARMUPS="${L2_HIGH_CTX_WARMUPS}" L2_LEG_LABEL="${label}" \
    L2_CACHE_K="${cache_k}" L2_CACHE_V="${cache_v}" L2_BIN="${bin}" \
    L2_OUT_JSON="${out_json}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python3 <<'PY'
import json, os, subprocess, time, urllib.request

port = int(os.environ["L2_PORT"])
n_predict = int(os.environ["L2_NUM_PREDICT"])
runs = max(1, int(os.environ["L2_BENCH_RUNS"]))
warmups = max(0, int(os.environ["L2_HIGH_CTX_WARMUPS"]))
url = f"http://127.0.0.1:{port}/completion"
prompt = (
    "List ten interesting facts about machine learning inference on NVIDIA CUDA. "
    "Number each fact.\n1."
)
visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"

def once(n: int) -> tuple[float, dict]:
    body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode())
    wall = time.perf_counter() - t0
    timings = data.get("timings") or {}
    tps = timings.get("predicted_per_second")
    if not isinstance(tps, (int, float)) or tps <= 0:
        pred_n = timings.get("predicted_n") or data.get("tokens_predicted") or n
        pred_ms = timings.get("predicted_ms")
        if isinstance(pred_ms, (int, float)) and pred_ms > 0:
            tps = float(pred_n) / (pred_ms / 1000.0)
        else:
            tps = float(pred_n) / max(wall, 1e-6)
    return float(tps), timings

def vram_mib():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"-i={visible}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        return int(out.strip().split()[0])
    except Exception:
        return None

for i in range(warmups):
    tps, _ = once(min(8, n_predict))
    print(f"  warmup {i}: {tps:.2f} tok/s")

vals = []
peak = 0
for i in range(runs):
    tps, _ = once(n_predict)
    vals.append(tps)
    used = vram_mib()
    if used is not None:
        peak = max(peak, used)
    print(f"  run {i}: {tps:.2f} tok/s vram={used}")

mean = sum(vals) / len(vals)
result = {
    "label": os.environ["L2_LEG_LABEL"],
    "llama_server_bin": os.environ["L2_BIN"],
    "cache_type_k": os.environ["L2_CACHE_K"],
    "cache_type_v": os.environ["L2_CACHE_V"],
    "decode_tok_per_s": round(mean, 2),
    "decode_tok_per_s_runs": [round(v, 2) for v in vals],
    "nvidia_smi_peak_mib": peak or None,
    "num_predict": n_predict,
    "warmups": warmups,
}
json.dump(result, open(os.environ["L2_OUT_JSON"], "w"), indent=2)
print(f"  mean={mean:.2f} tok/s peak_vram={peak} MiB")
PY
  kill "${_SERVER_PID}" 2>/dev/null || true
  _SERVER_PID=""
  fuser -k "${L2_PORT}/tcp" 2>/dev/null || true
  sleep 1
  _optional_llama_bench "${bin}" "${cache_k}" "${cache_v}" "${bench_log}"
}

IFS=',' read -r -a LEGS <<<"${L2_RQ_LEGS}"
LEG_JSONS=()
for leg in "${LEGS[@]}"; do
  leg="$(echo "${leg}" | tr -d '[:space:]')"
  [[ -z "${leg}" ]] && continue
  if ! _bench_leg "${leg}"; then
    echo "WARN: skipped/failed leg ${leg}" >&2
    continue
  fi
  LEG_JSONS+=("${_LEG_DIR}/${leg}.json")
done

export L2_OUT L2_NUM_CTX LLAMA_MODEL UNIFIED_BIN ROTOR_BIN L2_RQ_LEGS
export LEG_JSONS_CSV
LEG_JSONS_CSV="$(IFS=','; echo "${LEG_JSONS[*]}")"
python3 <<'PY'
import json, os
from datetime import datetime, timezone

paths = [p for p in os.environ.get("LEG_JSONS_CSV", "").split(",") if p and os.path.isfile(p)]
legs = {}
for p in paths:
    row = json.load(open(p))
    legs[row["label"]] = row

# Rank by decode; baseline = stock if present else first
baseline = "stock" if "stock" in legs else (next(iter(legs), None))
base = legs.get(baseline) or {}
bt = base.get("decode_tok_per_s")
bv = base.get("nvidia_smi_peak_mib")
ranking = []
for label, row in legs.items():
    entry = {
        "label": label,
        "cache": f"{row.get('cache_type_k')}/{row.get('cache_type_v')}",
        "decode_tok_per_s": row.get("decode_tok_per_s"),
        "nvidia_smi_peak_mib": row.get("nvidia_smi_peak_mib"),
    }
    t = row.get("decode_tok_per_s")
    v = row.get("nvidia_smi_peak_mib")
    if isinstance(bt, (int, float)) and isinstance(t, (int, float)) and bt > 0:
        entry["decode_vs_baseline_pct"] = round((t - bt) / bt * 100.0, 2)
    if isinstance(bv, int) and isinstance(v, int) and bv > 0:
        entry["vram_vs_baseline_pct"] = round((v - bv) / bv * 100.0, 2)
    ranking.append(entry)
ranking.sort(key=lambda e: (e.get("decode_tok_per_s") is None, -(e.get("decode_tok_per_s") or 0)))

# RotorQuant vs TBQ head-to-head if both present
h2h = {}
tbq = legs.get("tbq")
for rq in ("planar3", "iso3", "planar4", "iso4"):
    if rq not in legs or not tbq:
        continue
    rt, tt = legs[rq].get("decode_tok_per_s"), tbq.get("decode_tok_per_s")
    rv, tv = legs[rq].get("nvidia_smi_peak_mib"), tbq.get("nvidia_smi_peak_mib")
    row = {}
    if isinstance(rt, (int, float)) and isinstance(tt, (int, float)) and tt > 0:
        row["decode_vs_tbq_pct"] = round((rt - tt) / tt * 100.0, 2)
        row["beats_tbq_decode"] = rt > tt
    if isinstance(rv, int) and isinstance(tv, int) and tv > 0:
        row["vram_vs_tbq_pct"] = round((rv - tv) / tv * 100.0, 2)
        row["beats_tbq_vram"] = rv < tv
    h2h[rq] = row

out = {
    "ts": datetime.now(timezone.utc).isoformat(),
    "method": "direct llama-server /completion multi-leg",
    "unified_bin": os.environ.get("UNIFIED_BIN"),
    "rotorquant_bin": os.environ.get("ROTOR_BIN") or None,
    "gguf": os.environ["LLAMA_MODEL"],
    "num_ctx": int(os.environ["L2_NUM_CTX"]),
    "legs_requested": os.environ.get("L2_RQ_LEGS"),
    "baseline": baseline,
    "legs": legs,
    "ranking_by_decode": ranking,
    "rotorquant_vs_tbq": h2h,
    "doc": "docs/llama-fork-watchlist.md",
}
path = os.environ["L2_OUT"]
json.dump(out, open(path, "w"), indent=2)
print("")
print(f"wrote {path}")
print("ranking_by_decode:", json.dumps(ranking, indent=2))
if h2h:
    print("rotorquant_vs_tbq:", json.dumps(h2h, indent=2))
PY

echo "PASS: l2_rotorquant_ab (${L2_OUT})"
echo "Doc: docs/llama-fork-watchlist.md"
