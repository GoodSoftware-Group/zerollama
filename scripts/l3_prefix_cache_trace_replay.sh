#!/usr/bin/env bash
# Offline prefix cache trace replay (vLLM timed-trace inspired).
#
# WHY: replay committed JSONL decisions against KVCacheSpec without GPU smokes.
#
# Usage:
#   ./scripts/l3_prefix_cache_trace_replay.sh
#   TRACE=/path/to/trace.jsonl ./scripts/l3_prefix_cache_trace_replay.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}/runtime"

TRACE="${TRACE:-${ROOT}/runtime/tests/fixtures/prefix_cache_golden.jsonl}"
OUT="${L3_TRACE_REPLAY_OUT:-/tmp/l3-prefix-cache-trace-replay.json}"

echo "== prefix cache trace replay =="
PYTHONPATH=. python3 -m pytest \
  tests/test_prefix_cache_golden_trace.py \
  tests/test_prefix_cache_trace.py \
  tests/test_spec_bind.py \
  tests/test_decode_graph_policy.py \
  -q

echo "== replay trace file: ${TRACE} =="
PYTHONPATH=. python3 <<PY
import json
from pathlib import Path
from runtime.kv_cache_spec import KVCacheSpec
from runtime.prefix_cache_trace import replay_trace_file

path = Path("${TRACE}")
spec = KVCacheSpec(
    kind="sliding_window",
    effective_window=1024,
    allow_cache_prompt_base=True,
    allow_disk_persist=True,
    disk_ttl_ms=300000,
    speculative_draft=False,
)
mismatches = replay_trace_file(path, spec=spec)
report = {
    "trace": str(path),
    "lines": sum(1 for _ in path.open()),
    "mismatch_count": len(mismatches),
    "mismatches": [
        {"line": m.line, "field": m.field, "recorded": m.recorded, "expected": m.expected}
        for m in mismatches
    ],
    "pass": len(mismatches) == 0,
}
out = Path("${OUT}")
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
if mismatches:
    raise SystemExit(1)
PY

echo "PASS: l3_prefix_cache_trace_replay -> ${OUT}"
