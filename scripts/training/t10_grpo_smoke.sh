#!/usr/bin/env bash
# T10 GRPO scaffold smoke (host unit; no GPU).
#
# Always:
#   ./scripts/training/t10_grpo_smoke.sh
#     → unittest training_grpo validation + Go kind normalize
#
# Optional GPU (never production ports):
#   RUN_E2E_T10=1 CUDA_VISIBLE_DEVICES=1 ./scripts/training/t10_grpo_smoke.sh
#     → requires .venv-training with trl; submits kind=grpo dry_run via embed
#       (full GRPOTrainer needs grpo_run=true — lab only)
#
# Doc: docs/t10-grpo.md · ROADMAP T10
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -x "${ROOT}/.venv-training/bin/python" ]]; then
  PY="${ROOT}/.venv-training/bin/python"
elif [[ -n "${TRAINING_PYTHON:-}" && -x "${TRAINING_PYTHON}" ]]; then
  PY="${TRAINING_PYTHON}"
else
  PY=python3
fi

echo "== T10 unit: training_grpo validation =="
"$PY" -m unittest tests.test_training_grpo -v

echo "== T10 unit: Go kind normalize =="
if command -v go >/dev/null 2>&1; then
  # Full ./server test needs CGO + platform deps; CI runs this. Soft-skip on bare hosts.
  if (cd "$ROOT" && CGO_ENABLED=1 go test -count=1 ./server -run 'TestNormalizeTrainingJobKind' >/tmp/t10-go-kind.log 2>&1); then
    echo "OK: TestNormalizeTrainingJobKind"
  else
    echo "WARN: go test ./server not buildable here (see /tmp/t10-go-kind.log); CI covers kind normalize"
  fi
else
  echo "skip: go not installed"
fi

if [[ "${RUN_E2E_T10:-0}" != "1" ]]; then
  echo "OK: unit only (set RUN_E2E_T10=1 to re-check dry_run + trl advisory)"
  exit 0
fi

echo "== T10 lab: dry_run (no trl required) + optional grpo_run =="
"$PY" - <<'PY'
from training_grpo import process_grpo_request
res = process_grpo_request({
    "model_name": "sshleifer/tiny-gpt2",
    "prompts": ["hello"],
    "reward_builtin": "length",
})
print(res)
assert res.get("kind") == "grpo"
assert res.get("status") == "ok" and res.get("dry_run") is True
assert res.get("trained") is False
print("OK: dry_run trained=false")
if not res.get("trl"):
    print("note: trl missing — grpo_run would fail until pip install trl")
else:
    print("trl present — set grpo_run=true on a free GPU for Trainer lab")
PY
