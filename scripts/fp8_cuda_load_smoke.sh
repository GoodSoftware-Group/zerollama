#!/usr/bin/env bash
# Native FP8 GGUF CUDA load smoke — binary probe + llama-server /completion.
#
# WHY: docs/cuda-lanes.md P1 / docs/native-fp8-gguf.md. Proves a real FP8_E4M3
# (or E5M2) GGUF loads on packaged CUDA llama-server and decodes.
#
#   FP8_GGUF=/path/to/fp8.gguf ./scripts/fp8_cuda_load_smoke.sh
#
# Default fixture (this host):
#   /mnt/ssd2/models/fp8/tinyllama-fp8-e2e/tinyllama-1.1b-fp8_e4m3.gguf
#   (from nm-testing/TinyLlama-1.1B-Chat-v1.0-FP8-e2e via convert --fp8-native)
#
# Env: LLAMA_SERVER_BIN, CUDA_VISIBLE_DEVICES (default 1), L2_PORT (18083),
#      FP8_NUM_CTX (2048), FP8_NUM_PREDICT (32), FP8_OUT (/tmp/fp8-cuda-load-smoke.json)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/linux_runtime_serve_lib.sh
source "${ROOT}/scripts/linux_runtime_serve_lib.sh"

FP8_GGUF="${FP8_GGUF:-/mnt/ssd2/models/fp8/tinyllama-fp8-e2e/tinyllama-1.1b-fp8_e4m3.gguf}"
FP8_NUM_CTX="${FP8_NUM_CTX:-2048}"
FP8_NUM_PREDICT="${FP8_NUM_PREDICT:-32}"
L2_PORT="${L2_PORT:-18083}"
OUT="${FP8_OUT:-/tmp/fp8-cuda-load-smoke.json}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

if [[ ! -f "${FP8_GGUF}" ]]; then
  echo "Set FP8_GGUF to a native FP8 GGUF (missing: ${FP8_GGUF})" >&2
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

echo "== FP8 CUDA load smoke =="
"${ROOT}/scripts/fp8_cuda_probe.sh"

_SERVER_PID=""
cleanup() {
  [[ -n "${_SERVER_PID}" ]] && kill "${_SERVER_PID}" 2>/dev/null || true
  fuser -k "${L2_PORT}/tcp" 2>/dev/null || true
}
trap cleanup EXIT

LOG="$(mktemp /tmp/fp8-load-XXXXXX.log)"
fuser -k "${L2_PORT}/tcp" 2>/dev/null || true
sleep 1
echo "gguf=${FP8_GGUF}"
echo "bin=${LLAMA_SERVER_BIN}"
"${LLAMA_SERVER_BIN}" -m "${FP8_GGUF}" -c "${FP8_NUM_CTX}" -ngl 99 -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --host 127.0.0.1 --port "${L2_PORT}" -np 1 \
  >"${LOG}" 2>&1 &
_SERVER_PID=$!

ok=0
for _ in $(seq 1 90); do
  if curl -sf -m 3 "http://127.0.0.1:${L2_PORT}/health" >/dev/null 2>&1; then
    ok=1
    break
  fi
  if ! kill -0 "${_SERVER_PID}" 2>/dev/null; then
    echo "FAIL: llama-server died" >&2
    tail -50 "${LOG}" >&2
    exit 1
  fi
  sleep 1
done
if [[ "${ok}" -ne 1 ]]; then
  echo "FAIL: health timeout" >&2
  tail -50 "${LOG}" >&2
  exit 1
fi

FP8_GGUF="${FP8_GGUF}" L2_PORT="${L2_PORT}" FP8_NUM_PREDICT="${FP8_NUM_PREDICT}" \
  FP8_NUM_CTX="${FP8_NUM_CTX}" OUT="${OUT}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN}" ROOT="${ROOT}" python3 <<'PY'
import json, os, subprocess, sys, urllib.request
from pathlib import Path

port = int(os.environ["L2_PORT"])
n_predict = int(os.environ["FP8_NUM_PREDICT"])
gguf = os.environ["FP8_GGUF"]
out = Path(os.environ["OUT"])
root = Path(os.environ["ROOT"])
visible = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0].strip() or "0"
url = f"http://127.0.0.1:{port}/completion"
body = json.dumps({"prompt": "List three colors:", "n_predict": n_predict, "temperature": 0}).encode()
req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=120) as resp:
    data = json.loads(resp.read().decode())
timings = data.get("timings") or {}
content = (data.get("content") or "").strip()
pred_tps = timings.get("predicted_per_second")
if not isinstance(pred_tps, (int, float)) or pred_tps <= 0:
    raise SystemExit(f"FAIL: bad timings {timings}")
if not content:
    raise SystemExit("FAIL: empty completion content")

vram_mib = None
try:
    vram_mib = int(
        subprocess.check_output(
            ["nvidia-smi", f"-i={visible}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip().splitlines()[0]
    )
except Exception:
    pass

fp8_e4m3 = fp8_e5m2 = None
fp8_err = None
try:
    for cand in (
        root / "vendor" / "llama-cpp-8f114a9b" / "gguf-py",
        Path("/mnt/ssd2/zerollama-vendor/llama-cpp-8f114a9b/gguf-py"),
    ):
        if cand.is_dir():
            sys.path.insert(0, str(cand))
            break
    from gguf import GGUFReader
    r = GGUFReader(gguf)
    fp8_e4m3 = sum(1 for t in r.tensors if int(t.tensor_type) == 51)
    fp8_e5m2 = sum(1 for t in r.tensors if int(t.tensor_type) == 52)
except Exception as e:
    fp8_err = str(e)

artifact = {
    "ok": True,
    "gguf": gguf,
    "llama_server_bin": os.environ.get("LLAMA_SERVER_BIN"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "num_ctx": int(os.environ["FP8_NUM_CTX"]),
    "n_predict": n_predict,
    "content_preview": content[:160],
    "predicted_per_second": float(pred_tps),
    "prompt_per_second": timings.get("prompt_per_second"),
    "vram_used_mib": vram_mib,
    "fp8_e4m3_tensors": fp8_e4m3,
    "fp8_e5m2_tensors": fp8_e5m2,
    "fp8_count_error": fp8_err,
}
out.write_text(json.dumps(artifact, indent=2) + "\n")
print(json.dumps(artifact, indent=2))
print(f"PASS: wrote {out}")
PY

echo "PASS: fp8_cuda_load_smoke"
echo "Doc: docs/native-fp8-gguf.md"
