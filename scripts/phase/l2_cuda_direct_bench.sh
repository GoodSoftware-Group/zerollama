#!/usr/bin/env bash
# L2 CUDA direct A/B — stock q8_0 vs fork KV via llama-server /completion (no Python sidecar).
#
# WHY: l2_cuda_bench.sh (sidecar) under-reports absolute decode tok/s at long ctx on this
# host (stock ~5–32 tok/s via /api/generate vs ~290 via /completion at 131k). Use this
# script for quiet absolute decode + nvidia-smi VRAM; keep sidecar for runtime fork wiring.
#
#   CUDA_LLAMA_MODEL=/path/to.gguf L2_NUM_CTX=131072 ./scripts/phase/l2_cuda_direct_bench.sh
#
# Env:
#   CUDA_LLAMA_MODEL / LLAMA_MODEL — GGUF (required)
#   LLAMA_SERVER_BIN / LLAMA_CPP_ROOT — vendor or packaged binary
#   L2_NUM_CTX           — default 131072
#   L2_NUM_PREDICT       — default 64
#   L2_BENCH_RUNS        — timed runs after warmup (default 2)
#   L2_HIGH_CTX_WARMUPS  — short warmups (default 3)
#   L2_PORT              — llama-server listen (default 18082)
#   CUDA_VISIBLE_DEVICES — default 1 on multi-GPU (leave GPU0 for prod)
#   L2_SKIP_STOCK / L2_SKIP_FORK
#   L2_FORK_CACHE_TYPE_K/V — default tbq4_0 / tbq3_0 (vram profile)
#   L2_CUDA_DIRECT_OUT   — summary JSON (default /tmp/l2-cuda-direct-bench.json)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/runtime/linux_runtime_serve_lib.sh
source "${ROOT}/scripts/runtime/linux_runtime_serve_lib.sh"

LLAMA_MODEL="${CUDA_LLAMA_MODEL:-${LLAMA_MODEL:-}}"
if [[ -z "${LLAMA_MODEL}" || ! -f "${LLAMA_MODEL}" ]]; then
  echo "Set CUDA_LLAMA_MODEL to a GGUF path" >&2
  exit 1
fi

if [[ -n "${LLAMA_SERVER_BIN:-}" && -x "${LLAMA_SERVER_BIN}" ]]; then
  BIN="${LLAMA_SERVER_BIN}"
  UNIFIED_ROOT="$(cd "$(dirname "${BIN}")/../.." && pwd)"
elif [[ -n "${LLAMA_CPP_ROOT:-}" && -x "${LLAMA_CPP_ROOT}/build/bin/llama-server" ]]; then
  UNIFIED_ROOT="${LLAMA_CPP_ROOT}"
  BIN="${UNIFIED_ROOT}/build/bin/llama-server"
elif UNIFIED_ROOT="$(l1_vendor_llama_cpp_root "${ROOT}" 2>/dev/null)"; then
  BIN="${UNIFIED_ROOT}/build/bin/llama-server"
elif [[ -x /usr/local/lib/ollama/llama-server ]]; then
  BIN=/usr/local/lib/ollama/llama-server
  UNIFIED_ROOT=/usr/local/lib/ollama
else
  echo "Set LLAMA_SERVER_BIN or LLAMA_CPP_ROOT (built llama-server)" >&2
  exit 1
fi

export LLAMA_SERVER_BIN="${BIN}"
export LLAMA_CPP_ROOT="${UNIFIED_ROOT}"
linux_runtime_export_llama_ld_path

L2_NUM_CTX="${L2_NUM_CTX:-131072}"
L2_NUM_PREDICT="${L2_NUM_PREDICT:-64}"
L2_BENCH_RUNS="${L2_BENCH_RUNS:-2}"
L2_HIGH_CTX_WARMUPS="${L2_HIGH_CTX_WARMUPS:-3}"
L2_PORT="${L2_PORT:-18082}"
L2_OUT="${L2_CUDA_DIRECT_OUT:-/tmp/l2-cuda-direct-bench.json}"
# Prefer GPU1 when unset so dual-4090 prod on GPU0 is undisturbed.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
L2_FORK_CACHE_TYPE_K="${L2_FORK_CACHE_TYPE_K:-tbq4_0}"
L2_FORK_CACHE_TYPE_V="${L2_FORK_CACHE_TYPE_V:-tbq3_0}"

_PROMPT='List ten interesting facts about machine learning inference on NVIDIA CUDA. Number each fact.
1.'
_SERVER_PID=""
_LEG_DIR="$(mktemp -d /tmp/l2-cuda-direct-XXXXXX)"

cleanup() {
  [[ -n "${_SERVER_PID}" ]] && kill "${_SERVER_PID}" 2>/dev/null || true
  fuser -k "${L2_PORT}/tcp" 2>/dev/null || true
  rm -rf "${_LEG_DIR}"
}
trap cleanup EXIT

_start_server() {
  local cache_k="$1" cache_v="$2" log="$3"
  fuser -k "${L2_PORT}/tcp" 2>/dev/null || true
  sleep 1
  echo "starting ${BIN} -c ${L2_NUM_CTX} cache=${cache_k}/${cache_v} port=${L2_PORT}"
  "${BIN}" -m "${LLAMA_MODEL}" -c "${L2_NUM_CTX}" -ngl 99 -fa on \
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
      tail -40 "${log}" >&2
      return 1
    fi
    sleep 2
  done
  echo "llama-server health timeout; log:" >&2
  tail -40 "${log}" >&2
  return 1
}

_bench_leg() {
  local label="$1" cache_k="$2" cache_v="$3"
  local log="${_LEG_DIR}/${label}.log"
  local out_json="${_LEG_DIR}/${label}.json"
  echo ""
  echo "== L2 direct leg: ${label} cache=${cache_k}/${cache_v} =="
  _start_server "${cache_k}" "${cache_v}" "${log}"
  L2_PORT="${L2_PORT}" L2_NUM_PREDICT="${L2_NUM_PREDICT}" L2_BENCH_RUNS="${L2_BENCH_RUNS}" \
    L2_HIGH_CTX_WARMUPS="${L2_HIGH_CTX_WARMUPS}" L2_LEG_LABEL="${label}" \
    L2_CACHE_K="${cache_k}" L2_CACHE_V="${cache_v}" L2_OUT_JSON="${out_json}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python3 <<'PY'
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

def vram_mib() -> int | None:
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
    tps, tim = once(n_predict)
    vals.append(tps)
    used = vram_mib()
    if used is not None:
        peak = max(peak, used)
    print(f"  run {i}: {tps:.2f} tok/s vram={used}")

mean = sum(vals) / len(vals)
result = {
    "label": os.environ["L2_LEG_LABEL"],
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
}

STOCK_JSON=""
FORK_JSON=""

if [[ "${L2_SKIP_STOCK:-0}" != "1" ]]; then
  _bench_leg stock q8_0 q8_0
  STOCK_JSON="${_LEG_DIR}/stock.json"
fi
if [[ "${L2_SKIP_FORK:-0}" != "1" ]]; then
  _bench_leg fork "${L2_FORK_CACHE_TYPE_K}" "${L2_FORK_CACHE_TYPE_V}"
  FORK_JSON="${_LEG_DIR}/fork.json"
fi

export L2_OUT STOCK_JSON FORK_JSON L2_NUM_CTX LLAMA_MODEL BIN
python3 <<'PY'
import json, os
from datetime import datetime, timezone

out = {
    "ts": datetime.now(timezone.utc).isoformat(),
    "method": "direct llama-server /completion",
    "llama_server_bin": os.environ["BIN"],
    "gguf": os.environ["LLAMA_MODEL"],
    "num_ctx": int(os.environ["L2_NUM_CTX"]),
    "legs": {},
    "comparison": {},
}
stock_path = os.environ.get("STOCK_JSON") or ""
fork_path = os.environ.get("FORK_JSON") or ""
if stock_path and os.path.isfile(stock_path):
    out["legs"]["stock"] = json.load(open(stock_path))
if fork_path and os.path.isfile(fork_path):
    out["legs"]["fork"] = json.load(open(fork_path))

s = out["legs"].get("stock") or {}
f = out["legs"].get("fork") or {}
st = s.get("decode_tok_per_s")
ft = f.get("decode_tok_per_s")
sv = s.get("nvidia_smi_peak_mib")
fv = f.get("nvidia_smi_peak_mib")
comp = {}
if isinstance(st, (int, float)) and isinstance(ft, (int, float)) and st > 0:
    comp["stock_decode_tok_per_s"] = st
    comp["fork_decode_tok_per_s"] = ft
    comp["decode_delta_pct"] = round((ft - st) / st * 100.0, 2)
    comp["fork_wins_decode"] = ft > st
if isinstance(sv, int) and isinstance(fv, int) and sv > 0:
    comp["stock_nvidia_smi_peak_mib"] = sv
    comp["fork_nvidia_smi_peak_mib"] = fv
    comp["nvidia_smi_delta_pct"] = round((fv - sv) / sv * 100.0, 2)
    comp["fork_wins_vram"] = fv < sv
out["comparison"] = comp
path = os.environ["L2_OUT"]
json.dump(out, open(path, "w"), indent=2)
print("")
print(f"wrote {path}")
print("comparison:", json.dumps(comp, indent=2))
if comp.get("fork_wins_decode"):
    print("NOTE: fork won decode — unusual for TBQ; verify cache types and host load")
elif "decode_delta_pct" in comp:
    print(f"decode: stock wins (fork {comp['decode_delta_pct']:+.1f}%)")
if comp.get("fork_wins_vram"):
    print(f"vram: fork wins ({comp['nvidia_smi_delta_pct']:+.1f}%)")
PY

echo "PASS: l2_cuda_direct_bench (${L2_OUT})"
echo "Doc: docs/gpu-profiles-l2.md"
