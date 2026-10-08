#!/usr/bin/env bash
# L6 llama.cpp pin-debt ladder — read-only operator status.
#
# WHY: Enabler for bumping b10615 → … → b11351 without guessing Clef readiness.
# Does not start serve, rebuild vendor, or modify patches.
#
# Usage:
#   ./scripts/phase/l6_pin_ladder_status.sh
#   L6_PIN_OUT=/tmp/l6-pin-ladder.json ./scripts/phase/l6_pin_ladder_status.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
L6_PIN_OUT="${L6_PIN_OUT:-}"

VERSION="$(tr -d '[:space:]' < "${ROOT}/LLAMA_CPP_VERSION" 2>/dev/null || true)"
COMMIT="$(tr -d '[:space:]' < "${ROOT}/LLAMA_CPP_COMMIT" 2>/dev/null || true)"
FETCH_HEAD="$(grep -E '^FETCH_HEAD=' "${ROOT}/Makefile.sync" | cut -d= -f2 || true)"
VENDOR_DIR="${ROOT}/vendor/llama-cpp-${FETCH_HEAD:-b10615}"
CLEF_DIR="${ROOT}/llama/clef"
CLEF_PATCH="${CLEF_DIR}/upstream-002-clef.patch"
UPSTREAM_CLEF="${ROOT}/../ollama-upstream/llama/compat/002-clef.patch"
PATCH_COUNT="$(find "${ROOT}/llama/patches" -maxdepth 1 -name '*.patch' 2>/dev/null | wc -l | tr -d ' ')"

# Ladder rungs: tag|sha_prefix|note
LADDER=(
  "b10615|f280b26983ad0fdb705a0d9ebf0503e76f2899b0|rung 0 (rollback)"
  "b10864|5d806aa2575e01e126651fd69ab1ab6cefff861d|rung 1"

  "b10969|391fac16460f15233a7740550d858ac96df3419d|rung 2"
  "b11081|161755f29e415e2c33efe906e91843c068efd664|rung 3"
  "b11232|6f767fe960c3b97cf37fac4626c86400561ca1e4|rung 4 (ship / Clef floor)"
  "b11351|631109b34da437a3c4a5ebd75091d677671392e3|Ollama tip"
)

pin_ge_b11232=0
case "${VERSION}" in
  b1123*|b113*|b114*|b115*|b12*|b13*|b14*|b15*) pin_ge_b11232=1 ;;
esac

clef_head=0
[[ -f "${CLEF_DIR}/clef.cpp" && -f "${CLEF_DIR}/clef.h" ]] && clef_head=1
clef_convert=0
[[ -f "${CLEF_DIR}/convert.py" ]] && clef_convert=1
clef_deferred=0
[[ -f "${CLEF_PATCH}" ]] && clef_deferred=1
compat_clef_present=0
[[ -f "${ROOT}/llama/compat/002-clef.patch" ]] && compat_clef_present=1
clef_loader_skip=0
if grep -q 'add_skip_prefix(ml, "clef.")' "${ROOT}/llama/compat/llama-ollama-compat.cpp" 2>/dev/null; then
  clef_loader_skip=1
fi
clef_live_smoke=0
[[ -x "${ROOT}/scripts/phase/l6_clef_live_smoke.sh" || -f "${ROOT}/scripts/phase/l6_clef_live_smoke.sh" ]] && clef_live_smoke=1

clef_apply="skipped_no_vendor"
clef_wired_in_tree=0
if [[ -f "${VENDOR_DIR}/tools/server/server-context.cpp" ]] &&
   grep -q 'clef_head\|unique_ptr<clef_head>' "${VENDOR_DIR}/tools/server/server-context.cpp" 2>/dev/null; then
  clef_wired_in_tree=1
fi
_CLEF_APPLY_PATCH="${ROOT}/llama/compat/002-clef.patch"
[[ -f "${_CLEF_APPLY_PATCH}" ]] || _CLEF_APPLY_PATCH="${CLEF_PATCH}"
if [[ -d "${VENDOR_DIR}/.git" || -f "${VENDOR_DIR}/CMakeLists.txt" ]]; then
  if [[ -f "${_CLEF_APPLY_PATCH}" ]]; then
    if git -C "${VENDOR_DIR}" apply --check "${_CLEF_APPLY_PATCH}" >/dev/null 2>&1; then
      clef_apply="applies_clean"
    elif git -C "${VENDOR_DIR}" apply --check --reverse "${_CLEF_APPLY_PATCH}" >/dev/null 2>&1; then
      clef_apply="already_applied"
    else
      clef_apply="does_not_apply"
    fi
  else
    clef_apply="missing_deferred_patch"
  fi
fi
if [[ ${clef_wired_in_tree} -eq 1 && "${clef_apply}" == "does_not_apply" ]]; then
  clef_apply="wired_in_tree"
fi

echo "== L6 pin ladder status =="
echo "LLAMA_CPP_VERSION:     ${VERSION}"
echo "LLAMA_CPP_COMMIT:      ${COMMIT}"
echo "Makefile.sync FETCH:   ${FETCH_HEAD}"
echo "vendor dir:            ${VENDOR_DIR} ($([ -d "${VENDOR_DIR}" ] && echo present || echo missing))"
echo "llama/patches count:   ${PATCH_COUNT}"
echo ""
echo "Ladder (Ollama tip trail → b11351):"
for row in "${LADDER[@]}"; do
  IFS='|' read -r tag sha note <<<"${row}"
  mark="  "
  if [[ "${VERSION}" == "${tag}" ]] || [[ "${COMMIT}" == "${sha}" ]]; then
    mark="->"
  fi
  echo "  ${mark} ${tag}  ${sha:0:12}…  ${note}"
done
echo ""
echo "Clef staging:"
echo "  llama/clef head sources:     $([[ ${clef_head} -eq 1 ]] && echo yes || echo NO)"
echo "  llama/clef/convert.py:       $([[ ${clef_convert} -eq 1 ]] && echo yes || echo NO)"
echo "  deferred upstream patch:     $([[ ${clef_deferred} -eq 1 ]] && echo yes || echo NO) (${CLEF_PATCH})"
echo "  llama/compat/002-clef.patch: $([[ ${compat_clef_present} -eq 1 ]] && echo 'PRESENT (wire only at b11232+)' || echo 'absent (correct until b11232+)')"
echo "  loader skip clef.*:          $([[ ${clef_loader_skip} -eq 1 ]] && echo yes || echo NO)"
echo "  git apply --check (vendor):  ${clef_apply}"
echo "  server-context Clef wire:    $([[ ${clef_wired_in_tree} -eq 1 ]] && echo yes || echo no)"
echo "  pin ≥ b11232 (wire OK?):     $([[ ${pin_ge_b11232} -eq 1 ]] && echo yes || echo no)"
echo "  lab live smoke script:       $([[ ${clef_live_smoke} -eq 1 ]] && echo yes || echo NO)"
if [[ -f "${UPSTREAM_CLEF}" ]]; then
  echo "  ../ollama-upstream Clef:     present"
else
  echo "  ../ollama-upstream Clef:     missing (optional refresh source)"
fi
echo ""
echo "Doc: docs/llama-cpp-pin-ladder.md · runtime/LLAMA_CPP_PIN.md · llama/clef/README.md"
if [[ "${VERSION}" == "b11351" ]]; then
  echo "Next: Strands CUDA PointerRows (deferred) · tip prod live — rollback: l6_promote_tip_env.sh --rollback"
else
  echo "Next: bump one rung via docs/llama-cpp-pin-ladder.md#operator-runbook-one-rung"
fi

if [[ -n "${L6_PIN_OUT}" ]]; then
  VERSION="${VERSION}" COMMIT="${COMMIT}" FETCH_HEAD="${FETCH_HEAD}" \
    VENDOR_DIR="${VENDOR_DIR}" PATCH_COUNT="${PATCH_COUNT}" \
    clef_head="${clef_head}" clef_convert="${clef_convert}" \
    clef_loader_skip="${clef_loader_skip}" clef_live_smoke="${clef_live_smoke}" \
    clef_deferred="${clef_deferred}" \
    compat_clef_present="${compat_clef_present}" clef_apply="${clef_apply}" \
    pin_ge_b11232="${pin_ge_b11232}" L6_PIN_OUT="${L6_PIN_OUT}" python3 <<'PY'
import json, os, pathlib
report = {
    "ladder": "L6",
    "status": "tip_b11351" if os.environ.get("VERSION") == "b11351" else "enabler",
    "llama_cpp_version": os.environ.get("VERSION", ""),
    "llama_cpp_commit": os.environ.get("COMMIT", ""),
    "fetch_head": os.environ.get("FETCH_HEAD", ""),
    "vendor_dir": os.environ.get("VENDOR_DIR", ""),
    "vendor_present": pathlib.Path(os.environ.get("VENDOR_DIR", "")).is_dir(),
    "patch_count": int(os.environ.get("PATCH_COUNT") or 0),
    "clef_head_staged": os.environ.get("clef_head") == "1",
    "clef_convert_py": os.environ.get("clef_convert") == "1",
    "clef_loader_skip": os.environ.get("clef_loader_skip") == "1",
    "clef_live_smoke": os.environ.get("clef_live_smoke") == "1",
    "clef_deferred_patch": os.environ.get("clef_deferred") == "1",
    "compat_clef_auto_apply_present": os.environ.get("compat_clef_present") == "1",
    "clef_patch_apply_check": os.environ.get("clef_apply", ""),
    "pin_ge_b11232": os.environ.get("pin_ge_b11232") == "1",
    "tip_target": "b11351",
    "clef_floor": "b11232",
    "docs": [
        "docs/llama-cpp-pin-ladder.md",
        "runtime/LLAMA_CPP_PIN.md",
        "llama/clef/README.md",
        "docs/system-one-score.md",
    ],
}
path = pathlib.Path(os.environ["L6_PIN_OUT"])
path.write_text(json.dumps(report, indent=2) + "\n")
print(f"report: {path}")
PY
fi

echo ""
echo "PASS: l6_pin_ladder_status"
