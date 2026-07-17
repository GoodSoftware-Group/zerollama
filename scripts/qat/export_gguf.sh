#!/usr/bin/env bash
# Export a ternary-QAT HF checkpoint to GGUF (F16) and then to ternary GGUF
# quant formats (TQ1_0, TQ2_0) using mainline llama.cpp.
#
# Usage:
#   scripts/qat/export_gguf.sh <hf_checkpoint_dir> <output_name> [llama_cpp_dir]
#
# Example:
#   scripts/qat/export_gguf.sh out/qat/qwen25_0.5b_ternary qwen25_0.5b_ternary_qat
#
# Produces (relative to cwd, or GGUF_OUT_DIR if set):
#   <output_name>.F16.gguf
#   <output_name>.TQ1_0.gguf
#   <output_name>.TQ2_0.gguf
#
# Notes:
# - Builds mainline llama.cpp locally (Metal-enabled) if llama-quantize is not
#   already on PATH and not found in the given/default llama.cpp dir.
# - This is for the QAT-trained checkpoint's ternary weights, which are
#   already {-1,0,+1} * scale per group -- TQ1_0/TQ2_0 here just repack them.
#   For a naive PTQ "before" baseline, run this same TQ1_0 step directly on an
#   F16 GGUF of the *original, untrained* base model instead.

set -euo pipefail

CKPT_DIR="${1:?usage: export_gguf.sh <hf_checkpoint_dir> <output_name> [llama_cpp_dir]}"
OUT_NAME="${2:?usage: export_gguf.sh <hf_checkpoint_dir> <output_name> [llama_cpp_dir]}"
LLAMA_CPP_DIR="${3:-llama.cpp}"
OUT_DIR="${GGUF_OUT_DIR:-.}"

mkdir -p "$OUT_DIR"

echo "PROGRESS:5:locating llama-quantize / convert_hf_to_gguf.py"

find_llama_quantize() {
  if command -v llama-quantize >/dev/null 2>&1; then
    command -v llama-quantize
    return 0
  fi
  for candidate in \
    "$LLAMA_CPP_DIR/build/bin/llama-quantize" \
    "$LLAMA_CPP_DIR/build/llama-quantize"; do
    if [ -x "$candidate" ]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

LLAMA_QUANTIZE_BIN="$(find_llama_quantize || true)"

if [ -z "$LLAMA_QUANTIZE_BIN" ]; then
  echo "PROGRESS:10:llama-quantize not found, cloning + building mainline llama.cpp (Metal)"
  if [ ! -d "$LLAMA_CPP_DIR" ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
  fi
  cmake -B "$LLAMA_CPP_DIR/build" -S "$LLAMA_CPP_DIR" -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build "$LLAMA_CPP_DIR/build" --target llama-quantize llama-cli llama-bench -j "$(sysctl -n hw.ncpu)"
  LLAMA_QUANTIZE_BIN="$LLAMA_CPP_DIR/build/bin/llama-quantize"
fi

CONVERT_SCRIPT="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
if [ ! -f "$CONVERT_SCRIPT" ]; then
  echo "PROGRESS:12:convert_hf_to_gguf.py not found in $LLAMA_CPP_DIR, looking for a vendored copy"
  CONVERT_SCRIPT="$(find . -maxdepth 3 -iname convert_hf_to_gguf.py 2>/dev/null | head -1)"
  if [ -z "$CONVERT_SCRIPT" ]; then
    echo "error: could not find convert_hf_to_gguf.py anywhere" >&2
    exit 1
  fi
  echo "PROGRESS:13:using $CONVERT_SCRIPT"
fi

F16_GGUF="$OUT_DIR/${OUT_NAME}.F16.gguf"
TQ1_GGUF="$OUT_DIR/${OUT_NAME}.TQ1_0.gguf"
TQ2_GGUF="$OUT_DIR/${OUT_NAME}.TQ2_0.gguf"

echo "PROGRESS:20:converting $CKPT_DIR -> $F16_GGUF"
python3 "$CONVERT_SCRIPT" "$CKPT_DIR" --outfile "$F16_GGUF" --outtype f16

echo "PROGRESS:60:quantizing -> $TQ1_GGUF"
"$LLAMA_QUANTIZE_BIN" "$F16_GGUF" "$TQ1_GGUF" TQ1_0

echo "PROGRESS:80:quantizing -> $TQ2_GGUF"
"$LLAMA_QUANTIZE_BIN" "$F16_GGUF" "$TQ2_GGUF" TQ2_0

echo "PROGRESS:100:done. Outputs: $F16_GGUF $TQ1_GGUF $TQ2_GGUF"
