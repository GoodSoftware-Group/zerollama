#!/usr/bin/env bash
# Build + run BPE tokenize identity/bench against vocab-only GGUFs.
# WHY: prove patches 0106–0126 stay bit-identical to stock merge (FORCE_LEGACY) and
# optionally measure speedups without trusting throwaway /tmp benches.
# Does not start serve or touch production ports (11434/8081).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LLAMA_CPP="${LLAMA_CPP_ROOT:-$ROOT/../llama.cpp}"
BIN="${TMPDIR:-/tmp}/tokenize_bpe_identity_bench"
MODELS="${LLAMA_CPP}/models"

if [[ ! -f "${LLAMA_CPP}/include/llama.h" ]]; then
  echo "error: llama.cpp not found at ${LLAMA_CPP} (set LLAMA_CPP_ROOT)" >&2
  exit 1
fi
if [[ ! -f "${LLAMA_CPP}/build/bin/libllama.dylib" && ! -f "${LLAMA_CPP}/build/bin/libllama.so" ]]; then
  echo "error: build libllama first (cmake --build ${LLAMA_CPP}/build --target llama)" >&2
  exit 1
fi

# Sanity: patched tree must expose id-pair path
if ! grep -q 'has_bpe_id_pairs' "${LLAMA_CPP}/src/llama-vocab.cpp"; then
  echo "error: ${LLAMA_CPP} missing 0106 id-pair patch — apply llama/patches/0106-*.patch" >&2
  exit 1
fi
if ! grep -q 'pretok_cache' "${LLAMA_CPP}/src/llama-vocab.cpp"; then
  echo "error: ${LLAMA_CPP} missing 0107 pretok cache — apply llama/patches/0107-*.patch" >&2
  exit 1
fi
if ! grep -q 'unicode_cpt_append_utf8' "${LLAMA_CPP}/src/unicode.cpp"; then
  echo "error: ${LLAMA_CPP} missing 0108 pretok materialize — apply llama/patches/0108-*.patch" >&2
  exit 1
fi
if ! grep -qE 'ensure_text_collapsed|had_invalid_utf8' "${LLAMA_CPP}/src/unicode.cpp"; then
  echo "error: ${LLAMA_CPP} missing 0109 lazy collapse / byte-span — apply llama/patches/0109-*.patch" >&2
  exit 1
fi
if ! grep -q 'unicode_regex_split_qwen2_ascii_seg' "${LLAMA_CPP}/src/unicode.cpp"; then
  echo "error: ${LLAMA_CPP} missing 0110 ASCII pretok — apply llama/patches/0110-*.patch" >&2
  exit 1
fi
if ! grep -q 'LLAMA_BPE_FORCE_LEGACY_SPECIALS' "${LLAMA_CPP}/src/llama-vocab.cpp"; then
  echo "error: ${LLAMA_CPP} missing 0111 LTR specials — apply llama/patches/0111-*.patch" >&2
  exit 1
fi
if ! grep -q 'cache_specials_trie' "${LLAMA_CPP}/src/llama-vocab.cpp"; then
  echo "error: ${LLAMA_CPP} missing 0112 specials trie — apply llama/patches/0112-*.patch" >&2
  exit 1
fi
if ! grep -q 'llm_specials_byte_trie' "${LLAMA_CPP}/src/llama-vocab.cpp"; then
  echo "error: ${LLAMA_CPP} missing 0113 byte-indexed specials trie — apply llama/patches/0113-*.patch" >&2
  exit 1
fi
if ! grep -q 'unicode_regex_split_qwen2_ascii_bytes' "${LLAMA_CPP}/src/unicode.cpp"; then
  echo "error: ${LLAMA_CPP} missing 0114 Qwen2 ASCII byte pretok — apply llama/patches/0114-*.patch" >&2
  exit 1
fi
if ! grep -q 'unicode_regex_split_gpt2_ascii_bytes' "${LLAMA_CPP}/src/unicode.cpp"; then
  echo "error: ${LLAMA_CPP} missing 0115 GPT-2/Llama3 ASCII byte pretok — apply llama/patches/0115-*.patch" >&2
  exit 1
fi
if ! grep -q 'unicode_is_qwen35_regex_expr' "${LLAMA_CPP}/src/unicode.cpp"; then
  echo "error: ${LLAMA_CPP} missing 0116 Qwen3.5 ASCII byte pretok — apply llama/patches/0116-*.patch" >&2
  exit 1
fi
if ! grep -q 'LLAMA_BPE_NO_BYTE_ENC_FAST' "${LLAMA_CPP}/src/unicode.cpp"; then
  echo "error: ${LLAMA_CPP} missing 0117 byte-encode LUT — apply llama/patches/0117-*.patch" >&2
  exit 1
fi
if ! grep -q 'unicode_byte_enc_table' "${LLAMA_CPP}/src/unicode.cpp"; then
  echo "error: ${LLAMA_CPP} missing 0118 fuse materialize — apply llama/patches/0118-*.patch" >&2
  exit 1
fi
if ! grep -q 'unicode_regex_split_try_blob' "${LLAMA_CPP}/src/unicode.cpp"; then
  echo "error: ${LLAMA_CPP} missing 0119 pretok blob — apply llama/patches/0119-*.patch" >&2
  exit 1
fi
if ! grep -q 'pretok_cache_ready' "${LLAMA_CPP}/src/llama-vocab.cpp"; then
  echo "error: ${LLAMA_CPP} missing 0120 session pretok cache — apply llama/patches/0120-*.patch" >&2
  exit 1
fi
if ! grep -q 'unicode_fill_blob_from_cpt_offsets' "${LLAMA_CPP}/src/unicode.cpp"; then
  echo "error: ${LLAMA_CPP} missing 0121 general pretok blob — apply llama/patches/0121-*.patch" >&2
  exit 1
fi
if ! grep -q 'unicode_regex_split_qwen2_mixed_seg' "${LLAMA_CPP}/src/unicode.cpp"; then
  echo "error: ${LLAMA_CPP} missing 0122 ASCII islands — apply llama/patches/0122-*.patch" >&2
  exit 1
fi
if ! grep -q 'unicode_regex_split_gpt2_mixed_seg' "${LLAMA_CPP}/src/unicode.cpp"; then
  echo "error: ${LLAMA_CPP} missing 0123 GPT-2/Llama3/Qwen3.5 islands — apply llama/patches/0123-*.patch" >&2
  exit 1
fi
if ! grep -q 'LLAMA_BPE_NO_BYTE_MIXED' "${LLAMA_CPP}/src/unicode.cpp"; then
  echo "error: ${LLAMA_CPP} missing 0124 byte-mixed islands — apply llama/patches/0124-*.patch" >&2
  exit 1
fi
if ! grep -q 'unicode_word_is_space_plus_printable' "${LLAMA_CPP}/src/unicode.cpp"; then
  echo "error: ${LLAMA_CPP} missing 0125 space+printable encode — apply llama/patches/0125-*.patch" >&2
  exit 1
fi
if ! grep -q 'LLAMA_BPE_NO_SIMD_PRETOK' "${LLAMA_CPP}/src/unicode.cpp"; then
  echo "error: ${LLAMA_CPP} missing 0126 SIMD pretok consume — apply llama/patches/0126-*.patch" >&2
  exit 1
fi

clang++ -O2 -std=c++17 "${ROOT}/scripts/bench/tokenize_bpe_identity_bench.cpp" \
  -I"${LLAMA_CPP}/include" -I"${LLAMA_CPP}/ggml/include" \
  -L"${LLAMA_CPP}/build/bin" -lllama -Wl,-rpath,"${LLAMA_CPP}/build/bin" \
  -o "${BIN}"

vocabs=(
  ggml-vocab-qwen2.gguf
  ggml-vocab-qwen35.gguf
  ggml-vocab-gemma-4.gguf
  ggml-vocab-llama-bpe.gguf
  ggml-vocab-gpt-2.gguf
  ggml-vocab-falcon.gguf
  ggml-vocab-starcoder.gguf
  ggml-vocab-deepseek-coder.gguf
  ggml-vocab-command-r.gguf
  ggml-vocab-mpt.gguf
  ggml-vocab-refact.gguf
)
ec=0
BENCH_ARGS=()
if [[ "${1:-}" == "--bench" ]]; then
  BENCH_ARGS=(--bench)
fi
for v in "${vocabs[@]}"; do
  f="${MODELS}/${v}"
  if [[ ! -f "${f}" ]]; then
    echo "skip missing ${f}"
    continue
  fi
  echo "==== ${v} ===="
  if ! "${BIN}" "${f}" "${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"}"; then
    ec=1
  fi
done
exit "${ec}"
