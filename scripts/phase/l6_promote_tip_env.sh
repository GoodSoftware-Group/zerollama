#!/usr/bin/env bash
# Prepare (or print) tip b11351 llama-server env for the next production restart.
#
# WHY: CT production still runs LLAMA_SERVER_BIN=vendor/llama-cpp-b10615 while the
# repo pin is b11351 (Clef product e2e green on tip). This script never kills
# :8080 / :11434 / :8081 — operator restarts ~/bin/serve.sh after review.
#
# Usage:
#   ./scripts/phase/l6_promote_tip_env.sh              # dry-run checks + print export block
#   ./scripts/phase/l6_promote_tip_env.sh --write       # also write run/l6_tip_llama_server.env
#   ./scripts/phase/l6_promote_tip_env.sh --rollback    # print / write b10615 rollback env
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODE=promote
WRITE=0
for arg in "$@"; do
  case "${arg}" in
    --write) WRITE=1 ;;
    --rollback) MODE=rollback ;;
    -h|--help)
      sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "error: unknown arg ${arg}" >&2; exit 2 ;;
  esac
done

TIP_ROOT="${ROOT}/vendor/llama-cpp-b11351"
TIP_BIN="${TIP_ROOT}/build/bin/llama-server"
ROLL_ROOT="${ROOT}/vendor/llama-cpp-b10615"
ROLL_BIN="${ROLL_ROOT}/build/bin/llama-server"
OUT_ENV="${ROOT}/run/l6_tip_llama_server.env"
LAB_ZL="${ROOT}/run/zerollama-lab"

if [[ "${MODE}" == "promote" ]]; then
  TARGET_ROOT="${TIP_ROOT}"
  TARGET_BIN="${TIP_BIN}"
  LABEL="tip b11351"
else
  TARGET_ROOT="${ROLL_ROOT}"
  TARGET_BIN="${ROLL_BIN}"
  LABEL="rollback b10615"
fi

echo "== L6 ${LABEL} env =="
if [[ ! -x "${TARGET_BIN}" ]]; then
  echo "error: missing ${TARGET_BIN}" >&2
  echo "hint: CUDA_HOME=/usr/local/cuda-12.8 CMAKE_CUDA_ARCHITECTURES=120-real ./scripts/build/build_llama_server.sh" >&2
  exit 1
fi

# Tip wire smoke (Clef symbols live in server-impl, not always the thin wrapper).
if [[ "${MODE}" == "promote" ]]; then
  if ! rg -q 'ZEROLLAMA_CLEF_DIR|clef_head|score_fields' \
      "${TIP_ROOT}/tools/server/server-context.cpp" \
      "${TIP_ROOT}/tools/server/CMakeLists.txt" 2>/dev/null; then
    echo "error: tip vendor missing Clef server wire" >&2
    exit 1
  fi
  echo "Clef server wire:     yes"
  echo "tip llama-server:     ${TARGET_BIN}"
  "${TARGET_BIN}" --version 2>&1 | head -3 | sed 's/^/  /'
fi

# Lab Go binary (System One + ggufIsClef + tip CGO) — do not overwrite run/zerollama.
if [[ -x /tmp/zerollama-lab ]]; then
  mkdir -p "${ROOT}/run"
  cp -f /tmp/zerollama-lab "${LAB_ZL}"
  chmod +x "${LAB_ZL}"
  echo "lab Go binary:        ${LAB_ZL} (from /tmp/zerollama-lab)"
elif [[ -x "${LAB_ZL}" ]]; then
  echo "lab Go binary:        ${LAB_ZL} (existing)"
else
  echo "lab Go binary:        missing — rebuild: CGO_ENABLED=1 go build -o ${LAB_ZL} ."
fi

# Production currently holds VRAM — report only.
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU (read-only):"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/  /'
fi
if curl -sS --max-time 2 http://127.0.0.1:8080/api/version >/dev/null 2>&1; then
  echo "production :8080:     UP (this script will NOT restart it)"
else
  echo "production :8080:     down/unreachable"
fi

BLOCK=$(cat <<EOF
# ${LABEL} — source before next ~/bin/serve.sh restart (CT 1564 only)
export LLAMA_CPP_ROOT=${TARGET_ROOT}
export LLAMA_CPP_BIN=${TARGET_ROOT}/build/bin
export LLAMA_SERVER_BIN=${TARGET_BIN}
export LLAMA_CPP_LIB=${TARGET_ROOT}/build/bin/libllama.so
export LD_LIBRARY_PATH=${TARGET_ROOT}/build/bin:\${LD_LIBRARY_PATH:-}
# Optional: tip Go daemon with System One / Clef routing (lab or next serve binary)
# export ZEROLLAMA_BIN=${LAB_ZL}
EOF
)

echo ""
echo "--- export block ---"
echo "${BLOCK}"
echo "--- end ---"

if [[ "${WRITE}" -eq 1 ]]; then
  mkdir -p "$(dirname "${OUT_ENV}")"
  printf '%s\n' "${BLOCK}" >"${OUT_ENV}"
  echo ""
  echo "Wrote ${OUT_ENV}"
  echo "Operator restart (YOU run this — agent will not):"
  echo "  # screen -S ollama  (or your usual session)"
  echo "  set -a; source ${OUT_ENV}; set +a"
  echo "  ~/bin/serve.sh"
fi

echo ""
echo "Lab verify (non-production ports) before promote:"
echo "  CLEF_E2E_ZEROLLAMA=${LAB_ZL} ./scripts/phase/l6_clef_decisions_e2e.sh"
echo "  CLEF_E2E_PRODUCT_GGUF=/root/models/clef-flash-gguf/clef-flash-ollama-q8_0.gguf \\"
echo "    CLEF_E2E_ZEROLLAMA=${LAB_ZL} CLEF_E2E_MODEL=clef-flash-lab \\"
echo "    ./scripts/phase/l6_clef_decisions_e2e.sh"
echo ""
echo "PASS: l6_promote_tip_env (${MODE})"
