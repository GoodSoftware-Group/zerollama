#!/usr/bin/env bash
# Phase 17 vision smoke — Go → llama-server with mmproj / chat images.
#
# WHY: criterion 6 opt-in path must prove vision GGUF loads and chat+image generates
# through llama-server, not ggml llamarunner.
#
# Usage (opt-in live gate):
#   RUN_E2E_P17_VISION=1 ./scripts/phase17_llama_server_vision_smoke.sh
#   RUN_E2E_P17_VISION=1 P17_VISION_MODEL=llava:latest ./scripts/phase17_llama_server_vision_smoke.sh
#
# Env:
#   RUN_E2E_P17_VISION=1   — required to run (vision models are heavy)
#   P17_VISION_MODEL      — pulled tag (auto-picked smallest projector manifest when unset)
#   P17_VISION_GGUF       — optional explicit main GGUF blob path
#   P17_VISION_IMAGE      — JPEG/PNG for /api/chat images[] (default: mtmd test fixture)
#   LLAMA_SERVER_BIN      — llama-server binary (auto-discovered when unset)
#   P17_VISION_HOST       — serve host:port (default 127.0.0.1:11448)
#   P17_VISION_OUT        — JSON report (default /tmp/phase17-llama-server-vision-smoke.json)
#   P17_VISION_NUM_PREDICT — default 16
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime_smoke_lib.sh"

if [[ "${RUN_E2E_P17_VISION:-0}" != "1" ]]; then
  echo "Set RUN_E2E_P17_VISION=1 to run live vision llama-server gate" >&2
  echo "Example: RUN_E2E_P17_VISION=1 P17_VISION_MODEL=llava:latest $0" >&2
  exit 1
fi

P17_VISION_OUT="${P17_VISION_OUT:-/tmp/phase17-llama-server-vision-smoke.json}"
P17_VISION_NUM_PREDICT="${P17_VISION_NUM_PREDICT:-16}"
P17_VISION_IMAGE="${P17_VISION_IMAGE:-${ROOT}/llama/llama.cpp/tools/mtmd/test-1.jpeg}"

smoke_m3_resolve_vision_signoff_model

if [[ ! -f "${P17_VISION_IMAGE}" ]]; then
  echo "Vision test image not found: ${P17_VISION_IMAGE}" >&2
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

P17_VISION_HOST="${P17_VISION_HOST:-127.0.0.1:11448}"
export OLLAMA_HOST="${P17_VISION_HOST}"
export ZEROLLAMA_LLAMA_SERVER=1
export ZEROLLAMA_LEGACY_RUNNER=1
export ZEROLLAMA_RUNTIME=0
unset ZEROLLAMA_RUNTIME_URL

echo "== Phase 17 llama-server vision smoke =="
echo "model: ${P17_VISION_MODEL}"
echo "gguf: ${LLAMA_MODEL}"
echo "image: ${P17_VISION_IMAGE}"
echo "llama-server: ${LLAMA_SERVER_BIN}"
echo "host: ${P17_VISION_HOST}"
echo "out: ${P17_VISION_OUT}"

_port="${P17_VISION_HOST##*:}"
if command -v fuser >/dev/null 2>&1; then
  fuser -k "${_port}/tcp" 2>/dev/null || true
elif command -v lsof >/dev/null 2>&1; then
  lsof -ti ":${_port}" | xargs kill -9 2>/dev/null || true
fi
sleep 1

LOG="${P17_VISION_OUT%.json}.log"
"${BIN}" serve --llama-server-backend >"${LOG}" 2>&1 &
SERVE_PID=$!
trap 'kill "${SERVE_PID}" 2>/dev/null || true' EXIT

_deadline=$((SECONDS + 240))
while (( SECONDS < _deadline )); do
  if curl -sf "http://${P17_VISION_HOST}/api/tags" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "${SERVE_PID}" 2>/dev/null; then
    echo "zerollama serve exited early; log:" >&2
    tail -60 "${LOG}" >&2 || true
    exit 1
  fi
  sleep 2
done
if ! curl -sf "http://${P17_VISION_HOST}/api/tags" >/dev/null 2>&1; then
  echo "timeout waiting for serve on ${P17_VISION_HOST}" >&2
  tail -60 "${LOG}" >&2 || true
  exit 1
fi

export P17_VISION_HOST P17_VISION_MODEL LLAMA_MODEL P17_VISION_IMAGE P17_VISION_NUM_PREDICT P17_VISION_OUT LOG
python3 <<'PY'
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

host = os.environ["P17_VISION_HOST"]
model = os.environ["P17_VISION_MODEL"]
gguf = os.environ["LLAMA_MODEL"]
image_path = Path(os.environ["P17_VISION_IMAGE"])
n_predict = int(os.environ.get("P17_VISION_NUM_PREDICT", "16"))
out_path = Path(os.environ["P17_VISION_OUT"])
log_path = Path(os.environ["LOG"])

base = f"http://{host}"
img_b64 = base64.standard_b64encode(image_path.read_bytes()).decode()


def http_json(method, path, body=None, timeout=900.0):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"{base}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode(errors="replace") if e.fp else ""
        raise SystemExit(f"HTTP {e.code} {path}: {body_text[:1200]}") from e


payload = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": "Describe this image in one short sentence.",
            "images": [img_b64],
        }
    ],
    "stream": False,
    "options": {"num_predict": n_predict, "num_ctx": 4096},
}
t0 = time.perf_counter()
out = http_json("POST", "/api/chat", payload)
elapsed = time.perf_counter() - t0

msg = out.get("message") or {}
content = (msg.get("content") or msg.get("thinking") or "").strip()
if not out.get("done"):
    raise SystemExit(f"chat incomplete: {out!r}")
if not content:
    raise SystemExit(f"empty chat response: {out!r}")

log_text = ""
if log_path.is_file():
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
if "using llama-server subprocess for model" not in log_text:
    raise SystemExit(
        "serve log missing 'using llama-server subprocess for model' — "
        "vision model may have routed to ggml runner"
    )

ps = http_json("GET", "/api/ps")
running = ps.get("models") or []

report = {
    "backend": "llama-server",
    "modality": "vision",
    "model": model,
    "gguf": gguf,
    "image": str(image_path),
    "wall_s": round(elapsed, 3),
    "response_preview": content[:200],
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
echo "PASS: phase17_llama_server_vision_smoke (${P17_VISION_OUT})"
echo "Doc: docs/phase17-llama-server.md"
