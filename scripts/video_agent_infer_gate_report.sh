#!/usr/bin/env bash
# Video agent infer gate verdict from video_agent_infer_smoke.json.
#
# WHY: operators need a one-line PASS/FAIL like l3_gate_report.sh after live VLM smoke;
# JSON report holds main verdict + optional preproc leg verdict separately.
#
# Usage:
#   ./scripts/video_agent_infer_gate_report.sh /tmp/video-agent-infer-smoke.json
set -euo pipefail

if [[ $# -lt 1 || ! -f "$1" ]]; then
  echo "usage: $0 <video-agent-infer-smoke.json>" >&2
  exit 1
fi

python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
verdict = data.get("verdict") or "fail"
reason = data.get("reason") or "missing verdict"
m2 = data.get("turn2_metrics") or {}
m1 = data.get("turn1_metrics") or {}
lc = data.get("llama_cache") or {}
v1_cached = data.get("v1_cached_tokens")

print(f"verdict: {verdict}")
print(f"reason:  {reason}")
print(f"model:   {data.get('model')}")
print(f"cache_key: {data.get('cache_key')}")
print(f"llama_cache.enabled: {lc.get('enabled')}")
print(f"turn1 cached_prompt_tokens: {m1.get('cached_prompt_tokens')}")
print(f"turn2 cached_prompt_tokens: {m2.get('cached_prompt_tokens')} (L3 gate)")
print(f"turn2 video_tokens: {m2.get('video_tokens')}")
print(f"turn2 image_tokens: {m2.get('image_tokens')}")
print(f"v1 cached_tokens (advisory): {v1_cached}")
print(f"infer_backend (from log): {data.get('infer_backend')}")
log_checks = data.get("log_checks") or {}
if log_checks:
    print(f"log session_cache_hit: {log_checks.get('session_cache_hit')}")
    print(f"log vision_embed_session_cache_hit: {log_checks.get('vision_embed_session_cache_hit')}")
    print(f"log vision_embed_engine_ollama: {log_checks.get('vision_embed_engine_ollama')}")
    print(f"log vision_grid_hints: {log_checks.get('vision_grid_hints')}")
    print(f"log padded_runner_inject: {log_checks.get('padded_runner_inject')}")
    print(f"log preprocessed_layout_session_cache_hit: {log_checks.get('preprocessed_layout_session_cache_hit')}")
    print(f"log access_cached_prompt_tokens: {log_checks.get('access_cached_prompt_tokens')}")
    print(f"log access_video_tokens: {log_checks.get('access_video_tokens')}")
pre = data.get("preprocessed_infer")
if pre:
    print(f"preproc verdict: {pre.get('verdict')}")
    print(f"preproc turn2 cached_prompt_tokens: {(pre.get('turn2_metrics') or {}).get('cached_prompt_tokens')}")
    print(f"preproc turn2_cached_ok: {pre.get('turn2_cached_ok')}")
    print(f"preproc grid_thw: {pre.get('grid_thw')}")

if verdict == "fail":
    sys.exit(1)
if pre and pre.get("verdict") == "fail":
    sys.exit(1)
if pre and pre.get("verdict") == "soft":
    print("preproc SOFT PASS: turn2 cached_prompt_tokens below minimum", file=sys.stderr)
PY
