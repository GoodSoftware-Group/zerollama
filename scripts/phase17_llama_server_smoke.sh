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
#   P17_MODEL — pulled local tag (default: RUN_E2E_PROXY_MODEL from blob auto-pick)
#   LLAMA_SERVER_BIN — llama-server binary (auto-discovered when unset)
#   P17_OUT          — JSON report (default /tmp/phase17-llama-server-smoke.json)
#   P17_NUM_PREDICT  — default 8
#   P17_SERVE_EXTRA  — serve flags (default --llama-server-backend; empty with P17_LINUX_AUTO=1)
#   P17_LINUX_AUTO   — if 1, plain `zerollama serve` (Linux auto ZEROLLAMA_LLAMA_SERVER=auto)
#   P17_MODE         — report label (default llama-server)
#   P17_ASSERT_RUNTIME_OFF — if 1, fail when :8081 /health responds (edge smoke)
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

export P17_MODEL="${P17_MODEL:-${RUN_E2E_PROXY_MODEL:-}}"
if [[ -z "${P17_MODEL}" ]]; then
  echo "No pulled tag for blob ${LLAMA_MODEL}; pull a model or set P17_MODEL=your-tag:latest" >&2
  exit 1
fi

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

# Strip any http:// or https:// scheme so P17_HOST is always host:port.
# OLLAMA_HOST may arrive as "http://127.0.0.1:8080" from CI wrapper scripts.
_raw_host="${OLLAMA_HOST:-127.0.0.1:11434}"
_raw_host="${_raw_host#http://}"
_raw_host="${_raw_host#https://}"
HOST="${_raw_host}"
P17_HOST="${P17_HOST:-${HOST}}"
export OLLAMA_HOST="${P17_HOST}"
P17_LINUX_AUTO="${P17_LINUX_AUTO:-0}"
P17_ASSERT_RUNTIME_OFF="${P17_ASSERT_RUNTIME_OFF:-0}"

if [[ "${P17_LINUX_AUTO}" == "1" ]]; then
  P17_MODE="${P17_MODE:-linux-auto}"
  P17_SERVE_EXTRA=""
  unset ZEROLLAMA_LLAMA_SERVER
  unset ZEROLLAMA_EDGE
elif [[ "${P17_SERVE_EXTRA+x}" != "x" ]]; then
  P17_SERVE_EXTRA="--llama-server-backend"
fi
P17_MODE="${P17_MODE:-llama-server}"

if [[ "${P17_SERVE_EXTRA}" == *"--edge"* ]]; then
  export ZEROLLAMA_EDGE=1
  P17_MODE=edge
  P17_ASSERT_RUNTIME_OFF=1
elif [[ "${P17_LINUX_AUTO}" != "1" ]]; then
  export ZEROLLAMA_LLAMA_SERVER=1
fi
export ZEROLLAMA_LEGACY_RUNNER=1
export ZEROLLAMA_RUNTIME=0
export ZEROLLAMA_RUNTIME_EMBED=0
export ZEROLLAMA_RUNTIME_DARWIN_SIDECAR=0
unset ZEROLLAMA_RUNTIME_URL

echo "== Phase 17 llama-server smoke =="
echo "mode: ${P17_MODE}"
echo "serve: zerollama serve${P17_SERVE_EXTRA:+ ${P17_SERVE_EXTRA}}"
echo "model: ${P17_MODEL}"
echo "gguf: ${LLAMA_MODEL}"
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
if [[ -n "${P17_SERVE_EXTRA}" ]]; then
  # shellcheck disable=SC2086
  "${BIN}" serve ${P17_SERVE_EXTRA} >"${LOG}" 2>&1 &
else
  "${BIN}" serve >"${LOG}" 2>&1 &
fi
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

if [[ "${P17_ASSERT_RUNTIME_OFF}" == "1" ]]; then
  _embed_port="${ZEROLLAMA_RUNTIME_EMBED_PORT:-8081}"
  if curl -sf "http://127.0.0.1:${_embed_port}/health" >/dev/null 2>&1; then
    echo "runtime /health responded on :${_embed_port} but ${P17_MODE} expects runtime off" >&2
    exit 1
  fi
fi

export P17_HOST LLAMA_MODEL P17_MODEL P17_NUM_PREDICT P17_OUT P17_MODE P17_SERVE_EXTRA P17_LINUX_AUTO LOG
python3 <<'PY'
import json
import os
import time
import urllib.request
from pathlib import Path

host = os.environ["P17_HOST"]
model = os.environ["P17_MODEL"]
gguf = os.environ["LLAMA_MODEL"]
n_predict = int(os.environ.get("P17_NUM_PREDICT", "8"))
out_path = Path(os.environ["P17_OUT"])
mode = os.environ.get("P17_MODE", "llama-server")
serve_extra = os.environ.get("P17_SERVE_EXTRA", "--llama-server-backend")
log_path = Path(os.environ.get("LOG", ""))

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


prompt = "Say hello in one word.\nAssistant:"
payload = {
    "model": model,
    "prompt": prompt,
    "stream": False,
    "options": {"num_predict": n_predict, "num_ctx": 4096},
}
t0 = time.perf_counter()
out = http_json("POST", "/api/generate", payload)
elapsed = time.perf_counter() - t0

if not out.get("done"):
    raise SystemExit(f"generate incomplete: {out!r}")
content = (out.get("response") or out.get("content") or out.get("thinking") or "").strip()
if not content:
    raise SystemExit(f"empty response: {out!r}")

log_text = ""
if log_path.is_file():
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
if "using llama-server subprocess for model" not in log_text:
    raise SystemExit(
        "serve log missing 'using llama-server subprocess for model' — "
        "model may have routed to ggml runner"
    )

if mode == "linux-auto":
    status = http_json("GET", "/api/status")
    backend = (status.get("inference") or {}).get("backend") or {}
    if backend.get("llama_server") != "auto":
        raise SystemExit(f"linux auto: expected backend.llama_server=auto, got {backend!r}")
    if backend.get("gguf_path") != "llama-server":
        raise SystemExit(f"linux auto: expected backend.gguf_path=llama-server, got {backend!r}")

if mode == "edge":
    status = http_json("GET", "/api/status")
    backend = (status.get("inference") or {}).get("backend") or {}
    if not backend.get("edge"):
        raise SystemExit(f"edge: expected backend.edge true, got {backend!r}")
    if backend.get("llama_server") != "explicit":
        raise SystemExit(f"edge: expected backend.llama_server=explicit, got {backend!r}")
    if backend.get("runtime_chat") != "off":
        raise SystemExit(f"edge: expected backend.runtime_chat=off, got {backend!r}")
    if backend.get("gguf_path") != "llama-server":
        raise SystemExit(f"edge: expected backend.gguf_path=llama-server, got {backend!r}")
    version = http_json("GET", "/api/version")
    if version.get("edge_build") not in (True, False):
        raise SystemExit(f"edge: expected /api/version edge_build bool, got {version!r}")

ps = http_json("GET", "/api/ps")
running = ps.get("models") or []

report = {
    "backend": "llama-server",
    "mode": mode,
    "serve_extra": serve_extra,
    "runtime_chat": "off" if mode == "edge" else "smoke-off",
    "model": model,
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
