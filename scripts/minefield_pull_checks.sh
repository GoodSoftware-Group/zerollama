#!/usr/bin/env bash
# Fetch runnable checks from model-serving-minefield and run the ones that
# speak OpenAI /v1 against a *lab* zerollama.
#
# Refuses production :11434 / :8081.
#
#   ./scripts/minefield_pull_checks.sh qwen3:0.6b
#   ./scripts/minefield_pull_checks.sh qwen2.5:0.5b budget
#
# Env: BASE_URL (default http://127.0.0.1:11435/v1), WORK_DIR, CHECKS
# CHECKS subset: budget tokenize cache (default: all three)
set -euo pipefail

MODEL="${1:-}"
SHIFT="${2:-}"
[[ -n "${MODEL}" ]] || { echo "usage: $0 <model-tag> [budget|tokenize|cache|all]" >&2; exit 2; }
WHICH="${SHIFT:-all}"

BASE_URL="${BASE_URL:-http://127.0.0.1:11435/v1}"
WORK_DIR="${WORK_DIR:-/tmp/zerollama-minefield-lab/checks}"
RAW=https://raw.githubusercontent.com/Blackwellboy/model-serving-minefield/main/checks
mkdir -p "${WORK_DIR}"

case "${BASE_URL}" in
  *://127.0.0.1:11434*|*://localhost:11434*|*://127.0.0.1:8081*|*://localhost:8081*)
    echo "refusing production port in BASE_URL=${BASE_URL}" >&2
    exit 1
    ;;
esac

fetch() {
  local f="$1"
  curl -fsSL -o "${WORK_DIR}/${f}" "${RAW}/${f}"
}

run_ok() {
  local name="$1"; shift
  echo "=== ${name} ==="
  set +e
  "$@"
  local ec=$?
  set -e
  case $ec in
    0) echo "exit 0 CLEAN" ;;
    1) echo "exit 1 UNREACHABLE" ;;
    2) echo "exit 2 BLOCKING" ;;
    3) echo "exit 3 NOTHING_INSPECTED" ;;
    *) echo "exit ${ec}" ;;
  esac
}

echo "pull checks into ${WORK_DIR} base=${BASE_URL} model=${MODEL}"

want_budget=0 want_tok=0 want_cache=0
case "${WHICH}" in
  all) want_budget=1; want_tok=1; want_cache=1 ;;
  budget) want_budget=1 ;;
  tokenize) want_tok=1 ;;
  cache) want_cache=1 ;;
  *) echo "unknown check set: ${WHICH}" >&2; exit 2 ;;
esac

if [[ "$want_budget" == 1 ]]; then
  fetch reasoning_budget_probe.py
  # Small N/budget for lab smoke; raise for real eval floors.
  run_ok budget python3 "${WORK_DIR}/reasoning_budget_probe.py" \
    --base-url "${BASE_URL}" --model "${MODEL}" \
    --max-tokens "${MAX_TOKENS:-512}" -n "${N:-3}" --temp "${TEMP:-0.6}"
fi

if [[ "$want_tok" == 1 ]]; then
  fetch tokenized_length_assert.py
  run_ok tokenize python3 "${WORK_DIR}/tokenized_length_assert.py" \
    --base-url "${BASE_URL}" --model "${MODEL}" --target "${TARGET:-512}"
fi

if [[ "$want_cache" == 1 ]]; then
  fetch cache_hit_probe.py
  run_ok cache python3 "${WORK_DIR}/cache_hit_probe.py" \
    --base-url "${BASE_URL}" --model "${MODEL}" || true
fi

echo "done. Upstream checks are not vendored; re-fetch on each run."
