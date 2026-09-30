#!/usr/bin/env bash
# Syntax-check GLiNER2.5-Decide track scripts (CI-friendly, no model download).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fail=0

need_file() {
  local p="$1"
  if [[ ! -f "${ROOT}/${p}" ]]; then
    echo "missing: ${p}" >&2
    fail=1
  fi
}

need_exec() {
  local p="$1"
  need_file "${p}"
  if [[ -f "${ROOT}/${p}" && ! -x "${ROOT}/${p}" ]]; then
    echo "not executable: ${p}" >&2
    fail=1
  fi
}

need_file "docs/gliner-decide.md"
need_file "docs/gliner-decide-findings.md"
need_file "gliner/decide_server/app.py"
need_file "gliner/decide_server/requirements.txt"
need_file "agentskills/typed-decisions/SKILL.md"

SCRIPTS=(
  scripts/vendor/ensure_gliner_decide_model.sh
  scripts/serve/serve_gliner_decide_lab.sh
  scripts/smoke/gliner_decide_smoke.sh
  scripts/smoke/gliner_decide_go_e2e.sh
  scripts/check_gliner_decide_scripts.sh
)

for s in "${SCRIPTS[@]}"; do
  need_exec "${s}"
  if [[ -f "${ROOT}/${s}" ]]; then
    if ! bash -n "${ROOT}/${s}"; then
      echo "bash -n failed: ${s}" >&2
      fail=1
    fi
  fi
done

# Grep gates — Decide must not share NER URL.
grep -q 'ZEROLLAMA_GLINER_DECIDE_URL' "${ROOT}/envconfig/config.go" || { echo "missing GlinerDecideURL" >&2; fail=1; }
grep -q 'BackendGlinerDecide' "${ROOT}/types/model/modality.go" || { echo "missing BackendGlinerDecide" >&2; fail=1; }
grep -q '/v1/gliner-decide' "${ROOT}/server/routes.go" || { echo "missing route" >&2; fail=1; }
grep -q 'isGlinerDecideModelName' "${ROOT}/server/decisions_external_proxy.go" || { echo "missing decide name heuristic" >&2; fail=1; }

if [[ "${fail}" -ne 0 ]]; then
  echo "GLINER_DECIDE_SCRIPTS_CHECK_FAIL" >&2
  exit 1
fi
echo "GLINER_DECIDE_SCRIPTS_CHECK_OK"
