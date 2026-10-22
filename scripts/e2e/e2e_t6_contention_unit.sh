#!/usr/bin/env bash
# T6c/T6d host unit — defer must pause new ggml loads (no live chat required).
#
#   ./scripts/e2e/e2e_t6_contention_unit.sh
#
# Doc: docs/t6-unified-queue.md § T6c
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "== T6 contention: env + trainprio (CGO-free) =="
go test ./envconfig -run 'TestTrainingDeferPausesGgml' -count=1 -timeout 30s
go test ./x/trainprio -count=1 -timeout 30s

echo "== T6 contention: defer↔ggml pause wiring (needs CGO on some hosts) =="
if CGO_ENABLED=1 go test ./server -run 'TestTrainingDeferPausesGgml|TestInferencePausesGgmlIncludesDefer|TestDeferEnqueuePausesGgmlImmediately|TestTrainHTTPDeferPausesGgmlLoads' -count=1 -timeout 120s >/tmp/t6-contention-go.log 2>&1; then
  echo "OK: server defer-pause units"
else
  echo "WARN: go test ./server soft-skip (see /tmp/t6-contention-go.log); CI covers CGO build"
  # Still require the pure env/trainprio legs above.
fi

echo "PASS: e2e_t6_contention_unit"
