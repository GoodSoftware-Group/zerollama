#!/usr/bin/env bash
# Phase 17 smoke — Go → llama-server backend (upstream GGUF path).
#
# WHY: criterion 4/6 need E2E evidence that eligible plain text GGUF loads and
# generates through llama-server, not llamarunner/ollamarunner or Python runtime.
#
# Usage (Mac):
#   ./scripts/build_ollama_llama_server_darwin.sh
#   M3_LLAMA_MODEL=/path/to/small.gguf ./scripts/phase17_llama_server_smoke.sh
#
# Usage (Linux CUDA):
#   LLAMA_SERVER_BIN=/path/to/llama-server ./scripts/phase17_llama_server_smoke.sh
#   CUDA_LLAMA_MODEL=/path/to/model.gguf ./scripts/phase17_llama_server_smoke.sh
#
# Env:
#   M3_LLAMA_MODEL / CUDA_LLAMA_MODEL / LLAMA_MODEL — GGUF path (required)
#   LLAMA_SERVER_BIN — llama-server binary (auto-discovered when unset)
#   P17_OUT          — JSON report (default /tmp/phase17-llama-server-smoke.json)
#   P17_NUM_PREDICT  — default 8
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime_smoke_lib.sh"

P17_OUT="${P17_OUT:-/tmp/phase17-llama-server-smoke.json}"
P17_NUM_PREDICT="${P17_NUM_PREDICT:-8}"

if [[ -n "${CUDA_LLAMA_MODEL:-}" ]]; then
  export M3_LLAMA_MODEL="${CUDA_LLAMA_MODEL}"
fi
smoke_m3_resolve_signoff_model

BIN="${ROOT}/zerollama"
if [[ ! -x "${BIN}" ]]; then
  echo ">>> building zerollama" >&2
  if [[ "$(uname -s)" == "Darwin" ]]; then
    "${ROOT}/scripts/build_zerollama_mac.sh" "${BIN}"
  else
    (cd "${ROOT}" && go build -o "${BIN}" .)
  fi
fi

if [[ -z "${LLAMA_SERVER_BIN:-}" ]]; then
  for candidate in \
    "${ROOT}/build/llama-server-darwin/bin/llama-server" \
    "${ROOT}/../llama.cpp/build/bin/llama-server" \
    "${ROOT}/../ollama-upstream/build/llama-server-darwin/bin/llama-server"; do
    if [[ -x "${candidate}" ]]; then
      export LLAMA_SERVER_BIN="${candidate}"
      break
    fi
  done
fi
if [[ -z "${LLAMA_SERVER_BIN:-}" || ! -x "${LLAMA_SERVER_BIN}" ]]; then
  echo "Set LLAMA_SERVER_BIN or run ./scripts/build_ollama_llama_server_darwin.sh (Mac)" >&2
  exit 1
fi

HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
P17_HOST="${P17_HOST:-${HOST}}"
export OLLAMA_HOST="${P17_HOST}"
export ZEROLLAMA_LLAMA_SERVER=1
export ZEROLLAMA_LEGACY_RUNNER=1
export ZEROLLAMA_RUNTIME=0
unset ZEROLLAMA_RUNTIME_URL

echo "== Phase 17 llama-server smoke =="
echo "model: ${LLAMA_MODEL}"
echo "llama-server: ${LLAMA_SERVER_BIN}"
echo "host: ${P17_HOST}"
echo "out: ${P17_OUT}"

# Stop stale serve on the smoke port.
_port="${P17_HOST##*:}"
if command -v fuser >/dev/null 2>&1; then
  fuser -k "${_port}/tcp" 2>/dev/null || true
elif command -v lsof >/dev/null 2>&1; then
  lsof -ti ":${_port}" | xargs kill -9 2>/dev/null || true
fi
sleep 1

LOG="${P17_OUT%.json}.log"
"${BIN}" serve --llama-server-backend >"${LOG}" 2>&1 &
SERVE_PID=$!
trap 'kill "${SERVE_PID}" 2>/dev/null || true' EXIT

_deadline=$((SECONDS + 180))
while (( SECONDS < _deadline )); do
  if curl -sf "http://${P17_HOST}/api/tags" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "${SERVE_PID}" 2>/dev/null; then
    echo "zerollama serve exited early; log:" >&2
    tail -40 "${LOG}" >&2 || true
    exit 1
  fi
  sleep 2
done
if ! curl -sf "http://${P17_HOST}/api/tags" >/dev/null 2>&1; then
  echo "timeout waiting for serve on ${P17_HOST}" >&2
  tail -40 "${LOG}" >&2 || true
  exit 1
fi

export P17_HOST LLAMA_MODEL P17_NUM_PREDICT P17_OUT
python3 <<'PY'
import json
import os
import time
import urllib.request
from pathlib import Path

host = os.environ["P17_HOST"]
gguf = os.environ["LLAMA_MODEL"]
n_predict = int(os.environ.get("P17_NUM_PREDICT", "8"))
out_path = Path(os.environ["P17_OUT"])

base = f"http://{host}"


def http_json(method, path, body=None, timeout=600.0):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"{base}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


# Pull model into sched (creates from GGUF path via generate options).
prompt = "Say hello in one word.\nAssistant:"
payload = {
    "model": "phase17-smoke",
    "prompt": prompt,
    "stream": False,
    "options": {"gguf": gguf, "num_predict": n_predict},
}
t0 = time.perf_counter()
out = http_json("POST", "/api/generate", payload)
elapsed = time.perf_counter() - t0

if not out.get("done"):
    raise SystemExit(f"generate incomplete: {out!r}")
content = (out.get("response") or out.get("content") or "").strip()
if not content:
    raise SystemExit(f"empty response: {out!r}")

ps = http_json("GET", "/api/ps")
running = ps.get("models") or []

report = {
    "backend": "llama-server",
    "gguf": gguf,
    "wall_s": round(elapsed, 3),
    "response_preview": content[:120],
    "eval_count": out.get("eval_count"),
    "running_models": len(running),
    "pass": True,
}
out_path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
print(f"wrote {out_path}")
PY

kill "${SERVE_PID}" 2>/dev/null || true
trap - EXIT

echo ""
echo "PASS: phase17_llama_server_smoke (${P17_OUT})"
echo "Doc: docs/phase17-llama-server.md"
