#!/usr/bin/env bash
# T6 unified queue smoke — training defer, idle-wait, cross-queue FIFO (Go + runtime).
#
# Offline (always):
#   Go: defer queue, submit policy, cross-queue FIFO tickets
#   Python: cross_fifo, cross_queue_seq, defer admission
#
# Live (when serve + runtime respond):
#   GET /api/status inference.training.queue_policy
#   runtime /health go_coordination fifo fields
#
# Usage:
#   ./scripts/e2e_t6_queue_smoke.sh
#   OLLAMA_HOST=http://127.0.0.1:11434 ZEROLLAMA_RUNTIME_URL=http://127.0.0.1:8081 ./scripts/e2e_t6_queue_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:8080}"
export ZEROLLAMA_RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"

echo "== T6 Go unit tests (offline) =="
(
  cd "${ROOT}"
  go test ./server -run 'TestSubmitTrainingJob|TestTrainHTTPSubmit|TestDeferredTraining|TestDefer|TestTrainingDefer|TestCrossQueue|TestCrossFifo|TestSchedYield|TestPendingQueueOldest|TestAllocCrossQueue|TestShouldDefer|TestInferenceStatusTraining|TestStatusHandlerIncludesTraining' -count=1 -timeout 120s
)
(
  cd "${ROOT}"
  go test ./envconfig -run 'TestTrainingAllowedWindow' -count=1 -timeout 60s
)

echo ""
echo "== T6 runtime pytest (offline) =="
# shellcheck source=scripts/runtime_uv_venv.sh
source "${ROOT}/scripts/runtime_uv_venv.sh"
runtime_uv_venv
(
  cd "${ROOT}/runtime"
  PYTHONPATH=. "${RUNTIME_UV_PYTHON}" -m pytest \
    tests/test_cross_fifo.py \
    tests/test_cross_queue_seq.py \
    tests/test_defer_admission.py \
    -q
)

if curl -sf -m 5 "${OLLAMA_HOST%/}/api/status" >/dev/null 2>&1; then
  echo ""
  echo "== T6 /api/status queue_policy (live) =="
  status_json=$(curl -sf -m 10 "${OLLAMA_HOST%/}/api/status")
  export STATUS_JSON="$status_json"
  python3 <<'PY'
import json, os
d = json.loads(os.environ["STATUS_JSON"])
inf = d.get("inference") or {}
tr = inf.get("training") or {}
qp = tr.get("queue_policy") or {}
if not qp:
    raise SystemExit("missing inference.training.queue_policy on /api/status")
for k in ("wait_inference_idle", "wait_ggml_loaded", "wait_fail_closed",
          "queue_on_busy", "cross_queue_fifo"):
    if k not in qp:
        raise SystemExit(f"missing queue_policy.{k}")
print("queue_policy:", {k: qp[k] for k in sorted(qp) if qp.get(k) not in (False, 0, "", None)})
if qp.get("allowed_window"):
    print("allowed_window:", qp["allowed_window"])
print("ok: /api/status T6 fields")
PY
else
  echo "skip: live /api/status (serve not on ${OLLAMA_HOST})"
fi

if curl -sf -m 5 "${ZEROLLAMA_RUNTIME_URL%/}/health" >/dev/null 2>&1; then
  echo ""
  echo "== T6 runtime /health coordination (live) =="
  health_json=$(curl -sf -m 15 "${ZEROLLAMA_RUNTIME_URL%/}/health")
  export HEALTH_JSON="$health_json"
  python3 <<'PY'
import json, os
h = json.loads(os.environ["HEALTH_JSON"])
coord = h.get("go_coordination") or {}
ad = h.get("admission") or {}
back = (ad.get("backpressure") or {}).get("coordination") or {}
if coord:
    fifo_keys = [k for k in coord if k.startswith("fifo_")]
    if fifo_keys:
        print("go_coordination fifo:", {k: coord[k] for k in sorted(fifo_keys)})
    else:
        print("go_coordination present (no fifo keys yet — mirror may be empty at idle)")
elif back:
    print("admission.backpressure.coordination:", back.get("fresh", back))
else:
    print("warn: no go_coordination mirror (embedded-only runtime is ok)")
print("ok: runtime /health T6 coordination")
PY
else
  echo "skip: live runtime /health (not on ${ZEROLLAMA_RUNTIME_URL})"
fi

echo "PASS: e2e_t6_queue_smoke"
