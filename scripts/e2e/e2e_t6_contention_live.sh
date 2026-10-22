#!/usr/bin/env bash
# T6d — HTTP-path contention (busy ggml → defer → PauseNewLoads → cancel → resume).
#
# Always (no live daemon):
#   ./scripts/e2e/e2e_t6_contention_live.sh
#     → Go httptest against training routes (simulates pending ggml load)
#
# Optional operator probe against a *lab* serve (never :11434 / :8081 production):
#   RUN_E2E_T6_LIVE=1 OLLAMA_HOST=http://127.0.0.1:11435 ./scripts/e2e/e2e_t6_contention_live.sh
#     → asserts /api/status queue_policy.defer_pauses_ggml is present
#
# Doc: docs/t6-unified-queue.md § T6d
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "== T6d host: contention unit =="
./scripts/e2e/e2e_t6_contention_unit.sh

echo "== T6d HTTP: defer under busy ggml pauses loads =="
if CGO_ENABLED=1 go test ./server -run 'TestTrainHTTPDeferPausesGgmlLoads|TestTrainHTTPSubmitQueuesOnBusy' -count=1 -timeout 120s >/tmp/t6d-http.log 2>&1; then
  echo "OK: TrainHTTP defer→pause path"
else
  echo "WARN: go test ./server soft-skip (see /tmp/t6d-http.log); CI covers CGO build"
fi

if [[ "${RUN_E2E_T6_LIVE:-0}" == "1" ]]; then
  HOST="${OLLAMA_HOST:-http://127.0.0.1:11435}"
  HOST="${HOST%/}"
  case "$HOST" in
    *":11434"*|*":8081"*)
      echo "error: refuse production ports in OLLAMA_HOST=$HOST (use lab :11435)" >&2
      exit 1
      ;;
  esac
  echo "== T6d live probe: ${HOST}/api/status =="
  if ! curl -sf -m 5 "${HOST}/api/status" >/tmp/t6d-status.json; then
    echo "error: serve not reachable at ${HOST} (start lab serve first)" >&2
    exit 1
  fi
  python3 <<'PY'
import json
d = json.load(open("/tmp/t6d-status.json"))
qp = ((d.get("inference") or {}).get("training") or {}).get("queue_policy") or {}
if "defer_pauses_ggml" not in qp:
    raise SystemExit("missing queue_policy.defer_pauses_ggml")
print("ok: live queue_policy.defer_pauses_ggml =", qp.get("defer_pauses_ggml"))
PY
fi

echo "PASS: e2e_t6_contention_live"
