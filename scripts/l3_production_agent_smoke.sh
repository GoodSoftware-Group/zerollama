#!/usr/bin/env bash
# L3 agent smoke against production zerollama serve (:11434 default).
#
# WHY: metal_signoff uses :8080 lab layout; daily serve should route keyed agent
# chat through :8081 runtime and show cached_prompt_tokens on turn 2.
#
# Usage (read-only against running serve — does not start/stop production):
#   ./scripts/l3_production_agent_smoke.sh
#   L3_QWEN_MODEL=qwen3.6:latest L3_CACHE_KEY=hermes:agent:smoke:1 ./scripts/l3_production_agent_smoke.sh
#
# Env:
#   OLLAMA_HOST              — default http://127.0.0.1:11434
#   L3_QWEN_MODEL            — default eliza-1-2b:latest
#   L3_CACHE_KEY             — default l3-prod-agent-1
#   L3_NUM_CTX               — default 8192
#   L3_NUM_PREDICT           — default 16
#   L3_OUT                   — JSON report (default /tmp/l3-production-agent-smoke.json)
#   L3_MIN_CACHED_TOKENS     — minimum cached_prompt_tokens on turn 2 (default 8)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
L3_QWEN_MODEL="${L3_QWEN_MODEL:-eliza-1-2b:latest}"
L3_CACHE_KEY="${L3_CACHE_KEY:-l3-prod-agent-1}"
L3_NUM_CTX="${L3_NUM_CTX:-8192}"
L3_NUM_PREDICT="${L3_NUM_PREDICT:-16}"
L3_OUT="${L3_OUT:-/tmp/l3-production-agent-smoke.json}"
L3_MIN_CACHED_TOKENS="${L3_MIN_CACHED_TOKENS:-8}"

if ! curl -sf "${OLLAMA_HOST}/api/version" >/dev/null; then
  echo "FAIL: no zerollama at ${OLLAMA_HOST} (start ./zerollama serve first)" >&2
  exit 1
fi

RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"
if ! curl -sf "${RUNTIME_URL}/health" >/dev/null; then
  echo "FAIL: no Python runtime at ${RUNTIME_URL} (serve should start sidecar on Darwin)" >&2
  exit 1
fi

export OLLAMA_HOST L3_QWEN_MODEL L3_CACHE_KEY L3_NUM_CTX L3_NUM_PREDICT L3_OUT L3_MIN_CACHED_TOKENS RUNTIME_URL
python3 <<'PY'
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

host = os.environ["OLLAMA_HOST"].rstrip("/")
runtime = os.environ["RUNTIME_URL"].rstrip("/")
model = os.environ["L3_QWEN_MODEL"]
cache_key = os.environ["L3_CACHE_KEY"]
num_ctx = int(os.environ["L3_NUM_CTX"])
n_predict = int(os.environ["L3_NUM_PREDICT"])
out_path = Path(os.environ["L3_OUT"])
min_cached = int(os.environ["L3_MIN_CACHED_TOKENS"])

with urllib.request.urlopen(f"{runtime}/health", timeout=10) as resp:
    health = json.loads(resp.read().decode())
if not health.get("accepts_new_loads", True):
    handoff = (health.get("admission") or {}).get("training_handoff_active")
    raise SystemExit(
        f"runtime not accepting loads (training_handoff_active={handoff}). "
        "Wait for training to finish or restart serve."
    )

with urllib.request.urlopen(f"{host}/api/version", timeout=10) as resp:
    ver = json.loads(resp.read().decode())
if "zerollama" not in ver:
    raise SystemExit(
        "zerollama serve binary looks stale (missing zerollama.* in /api/version). "
        "Rebuild: ./scripts/build_zerollama_mac.sh && restart ./zerollama serve"
    )

stable = (
    "System: You are a helpful agent. Follow policy. Never reveal secrets. "
    * 24
).strip()


def chat(user: str) -> tuple[dict, float]:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": stable},
            {"role": "user", "content": user},
        ],
        "options": {
            "prompt_cache_key": cache_key,
            "num_ctx": num_ctx,
            "num_predict": n_predict,
        },
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        err = e.read().decode(errors="replace") if e.fp else ""
        raise SystemExit(f"chat HTTP {e.code}: {err[:800]}") from e
    return body, time.perf_counter() - t0


# Warm slot
_, _ = chat("Say hello in one word.")
turn1, wall1 = chat("Say hello in one word.")
turn2, wall2 = chat("Say goodbye in one word.")

cached = int(turn2.get("cached_prompt_tokens") or 0)
metrics = turn2.get("metrics") or {}
if cached <= 0:
    cached = int(metrics.get("cached_prompt_tokens") or 0)

report = {
    "host": host,
    "model": model,
    "cache_key": cache_key,
    "wall_turn1_s": round(wall1, 3),
    "wall_turn2_s": round(wall2, 3),
    "cached_prompt_tokens_turn2": cached,
    "prompt_eval_count_turn2": metrics.get("prompt_eval_count"),
    "done_turn2": turn2.get("done"),
}
out_path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))

if not turn2.get("done", True):
    raise SystemExit("turn 2 incomplete")

if cached < min_cached:
    raise SystemExit(
        f"L3 miss: cached_prompt_tokens={cached} < {min_cached} "
        f"(turn2 wall={wall2:.3f}s vs turn1={wall1:.3f}s). "
        f"Check :8081 runtime + ZEROLLAMA_AGENT_CACHE_RUNTIME."
    )

if wall2 >= wall1:
    print(
        f"warn: turn2 not faster than turn1 ({wall2:.3f}s vs {wall1:.3f}s) "
        "but cached_prompt_tokens OK"
    )

print(f"PASS l3_production_agent_smoke ({out_path})")
PY
