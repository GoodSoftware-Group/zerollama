#!/usr/bin/env bash
# Install / build Pure-C video-c (video-cli) and optional Wan GGUF/vocab helpers.
#
# Requires: uma_daemon running (does NOT start a second daemon).
# Usage:
#   ./scripts/video/install_wan_c.sh
#   ./scripts/video/install_wan_c.sh --ckpt-dir ~/.zerollama/third_party/wan/Wan2.1-T2V-1.3B
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VIDEO_C="$ROOT/x/video-c"
# Compat: x/wan-c → x/video-c
OUT_ROOT="${VIDEO_C_ROOT:-${WAN_C_ROOT:-$HOME/.zerollama/third_party/video-c}}"
CKPT="${1:-}"
if [[ "${1:-}" == "--ckpt-dir" ]]; then
  CKPT="${2:-}"
fi
if [[ -z "$CKPT" ]]; then
  CKPT="$HOME/.zerollama/third_party/wan/Wan2.1-T2V-1.3B"
fi

echo "==> check uma_daemon"
if command -v uma >/dev/null 2>&1; then
  if ! uma ping 2>/dev/null; then
    echo "ERROR: uma_daemon not responding. Start it first:" >&2
    echo "  cd \$BMTL/.../uma_toolkit && make uma-daemon" >&2
    echo "  (or open UMAStatus.app)" >&2
    exit 1
  fi
elif [[ -S /tmp/uma_daemon.sock ]]; then
  echo "socket present at /tmp/uma_daemon.sock"
else
  echo "ERROR: /tmp/uma_daemon.sock missing — start uma_daemon before video-cli." >&2
  exit 1
fi

echo "==> build video-cli"
make -C "$VIDEO_C"

mkdir -p "$OUT_ROOT"
cp -f "$VIDEO_C/video-cli" "$OUT_ROOT/video-cli"
chmod +x "$OUT_ROOT/video-cli"
ln -sfn video-cli "$OUT_ROOT/wan-cli"

if [[ -d "$CKPT" ]]; then
  echo "==> convert GGUF from $CKPT"
  python3 "$VIDEO_C/tools/convert_wan_to_gguf.py" --ckpt-dir "$CKPT" \
    -o "$OUT_ROOT/wan_t2v_1.3b.gguf" || {
    echo "WARN: GGUF convert failed (torch/safetensors?). Continue without." >&2
  }
  SPM="$CKPT/google/umt5-xxl/spiece.model"
  if [[ ! -f "$SPM" ]]; then
    SPM="$(find "$CKPT" -name 'spiece.model' 2>/dev/null | head -1 || true)"
  fi
  if [[ -n "${SPM:-}" && -f "$SPM" ]]; then
    echo "==> export tokenizer vocab from $SPM"
    python3 "$VIDEO_C/tools/export_umt5_spm.py" "$SPM" -o "$OUT_ROOT/umt5.vocab" || true
  fi
else
  echo "WARN: ckpt dir missing ($CKPT); skip GGUF/vocab"
fi

cat > "$OUT_ROOT/env.sh" <<EOF
export VIDEO_CLI="$OUT_ROOT/video-cli"
export WAN_CLI="$OUT_ROOT/video-cli"
export WAN_C_CKPT_DIR="$CKPT"
export WAN_C_GGUF="$OUT_ROOT/wan_t2v_1.3b.gguf"
export WAN_C_VOCAB="$OUT_ROOT/umt5.vocab"
EOF

echo "Installed video-cli → $OUT_ROOT"
echo "Source: source $OUT_ROOT/env.sh"
echo "Then register: ./scripts/video/register_wan_models.sh  (tag wan2.1-t2v-c:lab)"
echo "Or set ZEROLLAMA_VIDEO_CLI=\$VIDEO_CLI to force C on Python Wan tags."
echo "Clients still POST /v1/videos with a model tag only — no runner field."
