#!/usr/bin/env bash
# Fair A/B: upstream Ollama vs zerollama on Apple Silicon (M4-class).
#
# Usage:
#   ./scripts/phase/m4_upstream_vs_zerollama_bench.sh
#   MODEL=llama3.2:3b NUM_CTX=4096 EPOCHS=6 ./scripts/phase/m4_upstream_vs_zerollama_bench.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UP="${OLLAMA_UPSTREAM_DIR:-${ROOT}/../ollama-upstream}"
OUT="${M4_BENCH_OUT:-/tmp/m4-bench-$(date +%Y%m%d-%H%M%S)}"
MODEL="${MODEL:-llama3.2:3b}"
NUM_CTX="${NUM_CTX:-4096}"
EPOCHS="${EPOCHS:-6}"
MAX_TOKENS="${MAX_TOKENS:-200}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-${HOME}/.ollama/models}"

mkdir -p "${OUT}"
echo ">>> output: ${OUT}" >&2
echo ">>> model=${MODEL} num_ctx=${NUM_CTX} epochs=${EPOCHS}" >&2
echo ">>> hw: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)" >&2

stop_port() {
  local port="$1"
  local pids
  pids="$(lsof -ti tcp:"${port}" 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    kill ${pids} 2>/dev/null || true
    sleep 2
    kill -9 ${pids} 2>/dev/null || true
    sleep 1
  fi
}

wait_ready() {
  local url="$1"
  local log="$2"
  for _ in $(seq 1 60); do
    if curl -sf --max-time 2 "${url}/api/tags" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo ">>> server failed to start; tail ${log}:" >&2
  tail -30 "${log}" >&2 || true
  return 1
}

run_bench() {
  local host="$1"
  local label="$2"
  local outfile="${OUT}/${label}.csv"
  echo ">>> bench ${label} @ ${host}" >&2
  (
    cd "${ROOT}"
    go run ./cmd/bench \
      -host "${host}" \
      -model "${MODEL}" \
      -num-ctx "${NUM_CTX}" \
      -epochs "${EPOCHS}" \
      -max-tokens "${MAX_TOKENS}" \
      -format csv \
      -output "${outfile}"
  ) 2>"${OUT}/${label}.stderr"
  echo ">>> wrote ${outfile}" >&2
}

summarize() {
  python3 - "${OUT}" <<'PY'
import csv, glob, os, sys
out_dir = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(out_dir, "*.csv"))):
    label = os.path.splitext(os.path.basename(path))[0]
    with open(path, newline="") as f:
        gen = [r for r in csv.DictReader(f) if r.get("STEP") == "generate"]
    if not gen:
        continue
    speeds = [float(r["TOKEN_PER_SEC"]) for r in gen if r.get("TOKEN_PER_SEC")]
    if not speeds:
        continue
    avg = sum(speeds) / len(speeds)
    rows.append((label, avg, min(speeds), max(speeds), len(speeds)))

print(f"{'ARM':<28} {'AVG tok/s':>10} {'MIN':>10} {'MAX':>10} {'N':>3}")
print("-" * 65)
baseline = None
for label, avg, mn, mx, n in rows:
    if baseline is None:
        baseline = avg
    delta = ((avg / baseline) - 1.0) * 100.0 if baseline else 0.0
    sign = "+" if delta >= 0 else ""
    print(f"{label:<28} {avg:10.1f} {mn:10.1f} {mx:10.1f} {n:3d}  ({sign}{delta:.1f}% vs first)")
PY
}

# --- upstream ---
stop_port 11435
stop_port 11434
stop_port 11436

echo ">>> [1/3] upstream ollama :11435" >&2
OLLAMA_HOST=127.0.0.1:11435 "${UP}/ollama" serve >"${OUT}/upstream.log" 2>&1 &
UP_PID=$!
trap 'kill ${UP_PID} 2>/dev/null || true' EXIT
wait_ready "http://127.0.0.1:11435" "${OUT}/upstream.log"
run_bench "127.0.0.1:11435" "upstream-llama-server"
kill "${UP_PID}" 2>/dev/null || true
wait "${UP_PID}" 2>/dev/null || true
trap - EXIT
sleep 3

# --- zerollama ggml (Mac default) ---
echo ">>> [2/3] zerollama ggml Metal :11434" >&2
stop_port 11434
ZEROLLAMA_LEGACY_RUNNER=1 "${ROOT}/zerollama" serve >"${OUT}/zerollama-ggml.log" 2>&1 &
ZG_PID=$!
trap 'kill ${ZG_PID} 2>/dev/null || true' EXIT
wait_ready "http://127.0.0.1:11434" "${OUT}/zerollama-ggml.log"
run_bench "127.0.0.1:11434" "zerollama-ggml-metal"
kill "${ZG_PID}" 2>/dev/null || true
wait "${ZG_PID}" 2>/dev/null || true
trap - EXIT
sleep 3

# --- zerollama Go → llama-server (Phase 17) ---
LLAMA_BIN="${LLAMA_SERVER_BIN:-${UP}/build/llama-server-darwin/bin/llama-server}"
if [[ -x "${LLAMA_BIN}" ]]; then
  echo ">>> [3/3] zerollama --llama-server-backend :11434" >&2
  stop_port 11434
  LLAMA_SERVER_BIN="${LLAMA_BIN}" \
    "${ROOT}/zerollama" serve --llama-server-backend >"${OUT}/zerollama-llama-server.log" 2>&1 &
  ZL_PID=$!
  trap 'kill ${ZL_PID} 2>/dev/null || true' EXIT
  wait_ready "http://127.0.0.1:11434" "${OUT}/zerollama-llama-server.log"
  run_bench "127.0.0.1:11434" "zerollama-llama-server"
  kill "${ZL_PID}" 2>/dev/null || true
  wait "${ZL_PID}" 2>/dev/null || true
  trap - EXIT
else
  echo ">>> skip phase17 arm: no llama-server at ${LLAMA_BIN}" >&2
fi

echo "" >&2
echo "=== M4 benchmark summary (generate tok/s) ===" >&2
summarize | tee "${OUT}/summary.txt"
echo ">>> full results: ${OUT}" >&2
