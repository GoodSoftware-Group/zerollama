#!/usr/bin/env bash
# NVFP4 dual-4090 CUDA sign-off — binary probe + direct llama-server load/decode.
#
# WHY: docs/cuda-lanes.md P1. Proves NVFP4 GGUF loads on sm_89 (generic MMQ, not
# Blackwell MMA) and records decode tok/s + VRAM. Optional MXFP4 sibling A/B.
#
#   NVFP4_GGUF=/path/to/nvfp4.gguf ./scripts/nvfp4_cuda_signoff.sh
#   NVFP4_GGUF=... MXFP4_GGUF=... ./scripts/nvfp4_cuda_signoff.sh   # format A/B
#
# Defaults on this host:
#   NVFP4 → /mnt/ssd2/models/nvfp4/gpt-oss-20b/gpt-oss-20b-nvfp4.gguf
#   MXFP4 → /mnt/ssd2/models/nvfp4/gpt-oss-20b-mxfp4/gpt-oss-20b-mxfp4.gguf (if present)
#
# Env: LLAMA_SERVER_BIN, CUDA_VISIBLE_DEVICES (default 1), L2_PORT (18082),
#      NVFP4_NUM_CTX (8192), NVFP4_NUM_PREDICT (64), NVFP4_OUT (/tmp/nvfp4-cuda-signoff.json)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/linux_runtime_serve_lib.sh
source "${ROOT}/scripts/linux_runtime_serve_lib.sh"

NVFP4_GGUF="${NVFP4_GGUF:-/mnt/ssd2/models/nvfp4/gpt-oss-20b/gpt-oss-20b-nvfp4.gguf}"
MXFP4_GGUF="${MXFP4_GGUF:-/mnt/ssd2/models/nvfp4/gpt-oss-20b-mxfp4/gpt-oss-20b-mxfp4.gguf}"
NVFP4_NUM_CTX="${NVFP4_NUM_CTX:-8192}"
NVFP4_NUM_PREDICT="${NVFP4_NUM_PREDICT:-64}"
NVFP4_WARMUPS="${NVFP4_WARMUPS:-2}"
NVFP4_RUNS="${NVFP4_RUNS:-2}"
L2_PORT="${L2_PORT:-18082}"
OUT="${NVFP4_OUT:-/tmp/nvfp4-cuda-signoff.json}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

if [[ ! -f "${NVFP4_GGUF}" ]]; then
  echo "Set NVFP4_GGUF to an NVFP4 GGUF (missing: ${NVFP4_GGUF})" >&2
  exit 1
fi

if [[ -z "${LLAMA_CPP_ROOT:-}" ]]; then
  if root="$(l1_vendor_llama_cpp_root "${ROOT}" 2>/dev/null)"; then
    export LLAMA_CPP_ROOT="${root}"
  fi
fi
export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${LLAMA_CPP_ROOT:+${LLAMA_CPP_ROOT}/build/bin/llama-server}}"
[[ -x "${LLAMA_SERVER_BIN:-}" ]] || export LLAMA_SERVER_BIN=/usr/local/lib/ollama/llama-server
linux_runtime_export_llama_ld_path

echo "== NVFP4 CUDA sign-off =="
"${ROOT}/scripts/nvfp4_cuda_probe.sh"

_SERVER_PID=""
_LEG_DIR="$(mktemp -d /tmp/nvfp4-signoff-XXXXXX)"
cleanup() {
  [[ -n "${_SERVER_PID}" ]] && kill "${_SERVER_PID}" 2>/dev/null || true
  fuser -k "${L2_PORT}/tcp" 2>/dev/null || true
  rm -rf "${_LEG_DIR}"
}
trap cleanup EXIT

_bench_gguf() {
  local label="$1" gguf="$2"
  local log="${_LEG_DIR}/${label}.log"
  local json="${_LEG_DIR}/${label}.json"
  echo ""
  echo "== leg: ${label} =="
  echo "gguf=${gguf}"
  fuser -k "${L2_PORT}/tcp" 2>/dev/null || true
  sleep 1
  "${LLAMA_SERVER_BIN}" -m "${gguf}" -c "${NVFP4_NUM_CTX}" -ngl 99 -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --host 127.0.0.1 --port "${L2_PORT}" -np 1 -b 2048 -ub 512 \
    >"${log}" 2>&1 &
  _SERVER_PID=$!
  local i
  for i in $(seq 1 120); do
    if curl -sf -m 3 "http://127.0.0.1:${L2_PORT}/health" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "${_SERVER_PID}" 2>/dev/null; then
      echo "FAIL: llama-server died (${label})" >&2
      tail -50 "${log}" >&2
      return 1
    fi
    sleep 2
  done
  if ! curl -sf -m 3 "http://127.0.0.1:${L2_PORT}/health" >/dev/null 2>&1; then
    echo "FAIL: health timeout (${label})" >&2
    tail -50 "${log}" >&2
    return 1
  fi

  L2_PORT="${L2_PORT}" NVFP4_NUM_PREDICT="${NVFP4_NUM_PREDICT}" NVFP4_WARMUPS="${NVFP4_WARMUPS}" \
    NVFP4_RUNS="${NVFP4_RUNS}" LABEL="${label}" GGUF="${gguf}" OUT_JSON="${json}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" NVFP4_NUM_CTX="${NVFP4_NUM_CTX}" python3 <<'PY'
import json, os, subprocess, urllib.request

port = int(os.environ["L2_PORT"])
n_predict = int(os.environ["NVFP4_NUM_PREDICT"])
warmups = max(0, int(os.environ["NVFP4_WARMUPS"]))
runs = max(1, int(os.environ["NVFP4_RUNS"]))
url = f"http://127.0.0.1:{port}/completion"
prompt = "Explain NVFP4 vs MXFP4 in two short sentences.\n"
visible = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0].strip() or "0"

def once(n: int):
    body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode())
    timings = data.get("timings") or {}
    tps = timings.get("predicted_per_second")
    if not isinstance(tps, (int, float)) or tps <= 0:
        raise RuntimeError(f"bad timings: {timings}")
    content = (data.get("content") or "").strip()
    return float(tps), content, timings

def vram():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"-i={visible}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        return int(out.strip().split()[0])
    except Exception:
        return None

for i in range(warmups):
    tps, _, _ = once(min(16, n_predict))
    print(f"  warmup {i}: {tps:.2f} tok/s")

vals = []
peak = 0
sample = ""
for i in range(runs):
    tps, content, tim = once(n_predict)
    vals.append(tps)
    used = vram()
    if used is not None:
        peak = max(peak, used)
    if content:
        sample = content[:160]
    print(f"  run {i}: {tps:.2f} tok/s vram={used}")

mean = sum(vals) / len(vals)
if not sample:
    raise SystemExit("empty completion content")
result = {
    "label": os.environ["LABEL"],
    "gguf": os.environ["GGUF"],
    "num_ctx": int(os.environ["NVFP4_NUM_CTX"]),
    "num_predict": n_predict,
    "decode_tok_per_s": round(mean, 2),
    "decode_tok_per_s_runs": [round(v, 2) for v in vals],
    "nvidia_smi_peak_mib": peak or None,
    "sample": sample,
}
json.dump(result, open(os.environ["OUT_JSON"], "w"), indent=2)
print(f"  mean={mean:.2f} tok/s peak_vram={peak} MiB")
PY

  kill "${_SERVER_PID}" 2>/dev/null || true
  _SERVER_PID=""
  fuser -k "${L2_PORT}/tcp" 2>/dev/null || true
  sleep 1
}

_bench_gguf nvfp4 "${NVFP4_GGUF}"
if [[ -f "${MXFP4_GGUF}" ]]; then
  _bench_gguf mxfp4 "${MXFP4_GGUF}"
else
  echo "skip MXFP4 A/B (missing ${MXFP4_GGUF})"
fi

export OUT _LEG_DIR LLAMA_SERVER_BIN NVFP4_NUM_CTX
python3 <<'PY'
import json, os
from datetime import datetime, timezone

leg_dir = os.environ["_LEG_DIR"]
out = {
    "ts": datetime.now(timezone.utc).isoformat(),
    "lane": "dual_4090",
    "arch_note": "sm_89 generic MMQ (not Blackwell MMA)",
    "llama_server_bin": os.environ["LLAMA_SERVER_BIN"],
    "num_ctx": int(os.environ["NVFP4_NUM_CTX"]),
    "legs": {},
    "comparison": {},
}
for name in ("nvfp4", "mxfp4"):
    path = os.path.join(leg_dir, f"{name}.json")
    if os.path.isfile(path):
        out["legs"][name] = json.load(open(path))

n = out["legs"].get("nvfp4") or {}
m = out["legs"].get("mxfp4") or {}
comp = {}
if n and m and n.get("decode_tok_per_s") and m.get("decode_tok_per_s"):
    nt, mt = n["decode_tok_per_s"], m["decode_tok_per_s"]
    comp["nvfp4_decode_tok_per_s"] = nt
    comp["mxfp4_decode_tok_per_s"] = mt
    comp["nvfp4_vs_mxfp4_decode_delta_pct"] = round((nt - mt) / mt * 100.0, 2)
if n.get("nvidia_smi_peak_mib") and m.get("nvidia_smi_peak_mib"):
    nv, mv = n["nvidia_smi_peak_mib"], m["nvidia_smi_peak_mib"]
    comp["nvfp4_vram_mib"] = nv
    comp["mxfp4_vram_mib"] = mv
    comp["nvfp4_vs_mxfp4_vram_delta_pct"] = round((nv - mv) / mv * 100.0, 2)
out["comparison"] = comp
path = os.environ["OUT"]
json.dump(out, open(path, "w"), indent=2)
print("")
print(f"wrote {path}")
print(json.dumps(comp, indent=2))
PY

echo "PASS: nvfp4_cuda_signoff (${OUT})"
echo "Doc: docs/cuda-lanes.md"
