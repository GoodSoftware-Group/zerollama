#!/usr/bin/env bash
# DeepGEMM borrowings host unit — docs + non-goals (no clone, no GPU, no vendor).
#
#   ./scripts/phase/deepgemm_borrowings_unit.sh
#
# Doc: docs/deepgemm-borrowings.md · docs/deepgemm-borrowings-findings.md
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "== DeepGEMM: docs present =="
test -f docs/deepgemm-borrowings.md
test -f docs/deepgemm-borrowings-findings.md

echo "== DeepGEMM: non-goals still explicit =="
grep -q 'does \*\*not\*\* vendor DeepGEMM' docs/deepgemm-borrowings.md
grep -qi 'Ascend' docs/deepgemm-borrowings.md
grep -qi 'Mega MoE' docs/deepgemm-borrowings.md
grep -q 'SM90 / SM100' docs/deepgemm-borrowings-findings.md

echo "== DeepGEMM: no product import of deep_gemm =="
# Runtime / training must not depend on the DeepGEMM Python extension.
if rg -n --glob '!docs/**' --glob '!.cache/**' --glob '!vendor/**' \
  -e 'import deep_gemm|from deep_gemm|DeepGEMM-Ascend|torch_npu' \
  runtime training.py training_*.py server x 2>/dev/null | head -20; then
  echo "error: unexpected DeepGEMM / Ascend import in product paths" >&2
  exit 1
fi
echo "OK: no deep_gemm product imports"

echo "== DeepGEMM: AGENTS + cuda-lanes cross-links =="
grep -q 'deepgemm-borrowings.md' AGENTS.md
grep -q 'deepgemm-borrowings.md' docs/cuda-lanes.md

echo "PASS: deepgemm_borrowings_unit"
