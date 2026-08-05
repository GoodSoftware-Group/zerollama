#!/usr/bin/env bash
# Thin wrapper for training run_script → wan-cli (Pure-C Wan T2V).
# Emits PROGRESS: lines for the job queue.
set -euo pipefail

CLI="${WAN_CLI:-}"
if [[ -z "$CLI" || ! -x "$CLI" ]]; then
  echo "ERROR: WAN_CLI not set or not executable" >&2
  exit 1
fi

OUT="${WAN_OUTPUT_PATH:?WAN_OUTPUT_PATH required}"
CKPT="${WAN_CKPT_DIR:?WAN_CKPT_DIR required}"
PROMPT="${WAN_PROMPT:?WAN_PROMPT required}"
SIZE="${WAN_SIZE:-480*832}"
FRAMES="${WAN_FRAMES:-81}"
STEPS="${WAN_STEPS:-50}"
CFG="${WAN_CFG:-5.0}"
SHIFT="${WAN_SHIFT:-5.0}"
SEED="${WAN_SEED:-}"
VOCAB="${WAN_C_VOCAB:-}"
SOCK="${UMA_SOCK:-/tmp/uma_daemon.sock}"

W="${SIZE%%\**}"
H="${SIZE##*\*}"
if [[ -z "$W" || -z "$H" || "$W" == "$SIZE" ]]; then
  W=480
  H=832
fi

echo "PROGRESS: 0 starting wan-cli"

ARGS=(
  --ckpt-dir "$CKPT"
  --prompt "$PROMPT"
  --width "$W"
  --height "$H"
  --frames "$FRAMES"
  --steps "$STEPS"
  --cfg "$CFG"
  --shift "$SHIFT"
  --uma-sock "$SOCK"
  --out "$OUT"
)
[[ -n "$SEED" ]] && ARGS+=(--seed "$SEED")
[[ -n "$VOCAB" && -f "$VOCAB" ]] && ARGS+=(--vocab "$VOCAB")
[[ -n "${WAN_NEG_PROMPT:-}" ]] && ARGS+=(--negative-prompt "$WAN_NEG_PROMPT")

# Prefer local CPU stub only when explicitly requested (lab).
export UMA_WAN_LOCAL="${UMA_WAN_LOCAL:-0}"

echo "PROGRESS: 5 invoking $CLI"
"$CLI" "${ARGS[@]}"
echo "PROGRESS: 100 done"
