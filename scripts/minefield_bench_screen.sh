#!/usr/bin/env bash
# Minefield trap 110 — do not publish a single-stream bench taken on a shared endpoint.
#
# Refuses production :11434 / :8081. Warns when /api/ps already has a loaded
# runner (neighbours / prefix cache / another client's KV). Lab only.
#
#   ./scripts/minefield_bench_screen.sh
#   BASE_URL=http://127.0.0.1:11435 ./scripts/minefield_bench_screen.sh
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:11435}"
BASE_URL="${BASE_URL%/}"

case "${BASE_URL}" in
  *://127.0.0.1:11434*|*://localhost:11434*|*://127.0.0.1:8081*|*://localhost:8081*)
    echo "refusing production port in BASE_URL=${BASE_URL} (trap 110 — shared endpoint)" >&2
    exit 1
    ;;
esac

echo "bench screen base=${BASE_URL} (trap 110)"

python3 - "$BASE_URL" <<'PY'
import json, sys, urllib.request
base = sys.argv[1]
def get(path):
    with urllib.request.urlopen(base + path, timeout=10) as r:
        return json.load(r)

try:
    ver = get("/api/version")
except Exception as e:
    print(f"API not ready: {e} (trap 112 first — do not bench a dead listener)")
    sys.exit(2)

ps = get("/api/ps")
models = ps.get("models") or []
print(f"version={ver.get('version')!r} loaded_runners={len(models)}")
for m in models:
    name = m.get("name") or m.get("model") or "?"
    print(f"  warm={name}")

if models:
    print("Trap 110 SIGNAL: endpoint already has a loaded runner — single-stream tok/s measures neighbours, prefix cache, and that model's KV.")
    print("Use a dedicated lab serve, or unload / unique host before publishing.")
    sys.exit(3)

print("Trap 110: no warm runner on this lab base — still isolate the process (no other clients) before quoting tok/s.")
print("Related: trap 54 (run order), 48 (client vs server duration).")
PY
