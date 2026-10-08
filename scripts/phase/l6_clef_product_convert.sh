#!/usr/bin/env bash
# Convert Cloudflare/clef-flash HF → Ollama-wire GGUF (clef.* head + decision.type).
#
# WHY not ggml-org/Clef-Flash-GGUF: that pack uses native arch=clef + decision.*
# tensors; llama/clef/clef.cpp expects clef.* F32 + {backbone}.decision.{hidden_size,…}.
#
# Optional vision projector (CLEF_CONVERT_MMPROJ=1, default on):
#   → mmproj-*.gguf (clip / qwen3vl_merger) for Modelfile second FROM + /v1/decisions images.
#
# Lab only — does not touch production serve ports.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LLAMA_CPP="${CLEF_CONVERT_LLAMA_CPP:-${ROOT}/vendor/llama-cpp-b11351}"
SRC="${CLEF_CONVERT_SRC:-/root/models/clef-flash-hf}"
OUT="${CLEF_CONVERT_OUT:-/root/models/clef-flash-gguf/clef-flash-ollama-q8_0.gguf}"
OUTTYPE="${CLEF_CONVERT_OUTTYPE:-q8_0}"
MMPROJ="${CLEF_CONVERT_MMPROJ:-1}"
MMPROJ_OUT="${CLEF_CONVERT_MMPROJ_OUT:-$(dirname "${OUT}")/mmproj-clef-flash-f16.gguf}"

if [[ ! -d "${SRC}" ]]; then
  echo "error: missing HF tree ${SRC}" >&2
  echo "hint: hf download Cloudflare/clef-flash --local-dir ${SRC}" >&2
  exit 1
fi

python3 "${ROOT}/llama/clef/convert.py" --llama-cpp "${LLAMA_CPP}" --check-only "${SRC}"

mkdir -p "$(dirname "${OUT}")"
if [[ "${CLEF_CONVERT_SKIP_TEXT:-0}" != "1" ]]; then
  python3 "${ROOT}/llama/clef/convert.py" --llama-cpp "${LLAMA_CPP}" \
    "${SRC}" --outtype "${OUTTYPE}" --outfile "${OUT}"

  python3 - <<PY
from gguf import GGUFReader
r = GGUFReader("${OUT}")
arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
dtype = bytes(r.fields[f"{arch}.decision.type"].parts[-1]).decode()
assert dtype == "clef", (arch, dtype)
n = sum(1 for t in r.tensors if t.name.startswith("clef."))
assert n > 0, "no clef.* tensors"
print(f"OK: {arch}.decision.type=clef clef_tensors={n} -> ${OUT}")
PY
fi

if [[ "${MMPROJ}" == "1" ]]; then
  python3 "${ROOT}/llama/clef/convert.py" --llama-cpp "${LLAMA_CPP}" \
    "${SRC}" --mmproj --outtype f16 --outfile "${MMPROJ_OUT}"
  python3 - <<PY
from gguf import GGUFReader
r = GGUFReader("${MMPROJ_OUT}")
arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
assert arch == "clip", arch
assert "clip.vision.block_count" in r.fields
print(f"OK: mmproj arch={arch} tensors={len(r.tensors)} -> ${MMPROJ_OUT}")
PY
fi
