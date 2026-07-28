#!/usr/bin/env bash
# mlx_prefix_cache_smoke.sh — two-turn MLX prefix cache smoke for Gemma4 agents.
#
# Defaults to lab port :11435 so production Hermes (:11434) does not share the MLX
# runner or prefix trie with this test.
#
# Temp-0 / reproducibility (minefield traps 91–92): temperature=0 here is a
# sampling convenience for stable short replies, NOT a claim of bit-identical
# outputs across runs. Prefix-cache state (prompt_cache_key) intentionally
# carries across turns; do not use this script to assert deterministic decode
# without isolating cache and stating a prompt-length regime. See
# docs/model-serving-minefield.md and docs/testing-smoke.md.
#
#   ./scripts/mlx/mlx_prefix_cache_smoke.sh
#   MLX_SMOKE_START_SERVE=1 ./scripts/mlx/mlx_prefix_cache_smoke.sh
#   BASE_URL=http://127.0.0.1:11434 ./scripts/mlx/mlx_prefix_cache_smoke.sh   # explicit prod port
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SMOKE_HOST="${OLLAMA_HOST:-127.0.0.1:11435}"
SMOKE_HOST="${SMOKE_HOST#http://}"
SMOKE_HOST="${SMOKE_HOST#https://}"
BASE="${BASE_URL:-http://${SMOKE_HOST}}"
MODEL="${MODEL:-gemma4:26b-optiq}"
KEY="${PROMPT_CACHE_KEY:-mlx-smoke-$(date +%s)}"
MIN_CACHED="${MIN_CACHED_TOKENS:-4000}"
MAX_T2_SEC="${MAX_TURN2_SEC:-90}"

_started_serve=0
_smoke_pid=""

_cleanup() {
  if [[ "${_started_serve}" -eq 1 && -n "${_smoke_pid}" ]]; then
    kill "${_smoke_pid}" 2>/dev/null || true
    wait "${_smoke_pid}" 2>/dev/null || true
  fi
}

_ensure_serve() {
  if curl -sf -m 5 "${BASE}/api/version" >/dev/null 2>&1; then
    echo "smoke serve: ${BASE}"
    return 0
  fi
  if [[ "${MLX_SMOKE_START_SERVE:-}" != 1 ]]; then
    echo "error: no serve on ${BASE}" >&2
    echo "hint: Hermes uses :11434 — start an isolated smoke serve, e.g." >&2
    echo "  OLLAMA_HOST=127.0.0.1:11435 ./zerollama serve" >&2
    echo "  MLX_SMOKE_START_SERVE=1 ./scripts/mlx/mlx_prefix_cache_smoke.sh" >&2
    exit 1
  fi

  local bin="${ZEROLLAMA_BIN:-${ROOT}/zerollama}"
  if [[ ! -x "${bin}" ]]; then
    echo "error: zerollama binary not found at ${bin}" >&2
    exit 1
  fi

  export OLLAMA_HOST="${SMOKE_HOST}"
  echo "starting isolated smoke serve on ${BASE} ..."
  "${bin}" serve >/tmp/zerollama-mlx-smoke.log 2>&1 &
  _smoke_pid=$!
  _started_serve=1
  trap _cleanup EXIT

  local i
  for i in $(seq 1 45); do
    if curl -sf -m 5 "${BASE}/api/version" >/dev/null 2>&1; then
      echo "smoke serve ready: ${BASE} (log: /tmp/zerollama-mlx-smoke.log)"
      return 0
    fi
    if ! kill -0 "${_smoke_pid}" 2>/dev/null; then
      tail -20 /tmp/zerollama-mlx-smoke.log >&2 || true
      echo "error: smoke serve exited before /api/version was ready" >&2
      exit 1
    fi
    sleep 1
  done
  tail -20 /tmp/zerollama-mlx-smoke.log >&2 || true
  echo "error: timed out waiting for smoke serve on ${BASE}" >&2
  exit 1
}

_ensure_serve

python3 - "$BASE" "$MODEL" "$KEY" "$MIN_CACHED" "$MAX_T2_SEC" <<'PY'
import json, sys, time, urllib.request

base, model, key, min_cached_s, max_t2_s = sys.argv[1:6]
min_cached = int(min_cached_s)
max_t2 = float(max_t2_s)

def post(messages, label):
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": 16,
        "temperature": 0,
        "prompt_cache_key": key,
        "keep_alive": "30m",
        "options": {"num_ctx": 131072},
    }
    t0 = time.time()
    with urllib.request.urlopen(
        urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        ),
        timeout=600,
    ) as resp:
        out = json.load(resp)
    elapsed = time.time() - t0
    u = out.get("usage") or {}
    ptd = u.get("prompt_tokens_details") or {}
    cached = ptd.get("cached_tokens") or 0
    prompt = u.get("prompt_tokens") or 0
    reply = out["choices"][0]["message"]["content"]
    print(f"{label}: elapsed={elapsed:.1f}s prompt={prompt} cached={cached}")
    return reply, elapsed, cached, prompt

block = ("The quick brown fox jumps over the lazy dog. " * 8 + "\n") * 80
system = "You are a concise assistant.\n\n" + block
m1 = [{"role": "system", "content": system}, {"role": "user", "content": "Say hi briefly."}]
r1, t1_elapsed, t1_cached, t1_prompt = post(m1, "turn1")
m2 = m1 + [{"role": "assistant", "content": r1}, {"role": "user", "content": "Say bye briefly."}]
_, t2_elapsed, t2_cached, t2_prompt = post(m2, "turn2")

failures = []
if t2_cached < min_cached:
    failures.append(f"turn2 cached={t2_cached} < min={min_cached}")
if t2_elapsed > max_t2:
    failures.append(f"turn2 elapsed={t2_elapsed:.1f}s > max={max_t2}s")

if failures:
    print("FAIL: mlx prefix cache smoke key=" + key)
    for f in failures:
        print("  -", f)
    print(f"  turn1: prompt={t1_prompt} cached={t1_cached} elapsed={t1_elapsed:.1f}s")
    print(f"  turn2: prompt={t2_prompt} cached={t2_cached} elapsed={t2_elapsed:.1f}s")
    sys.exit(1)

print(f"PASS: mlx prefix cache smoke key={key} turn2_cached={t2_cached} turn2_elapsed={t2_elapsed:.1f}s")
PY
