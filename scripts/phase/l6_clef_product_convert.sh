#!/usr/bin/env bash
# Convert Cloudflare/clef-flash HF → Ollama-wire GGUF (clef.* head + decision.type).
#
# WHY not ggml-org/Clef-Flash-GGUF: that pack uses native arch=clef + decision.*
# tensors; llama/clef/clef.cpp expects clef.* F32 + {backbone}.decision.{hidden_size,…}.
#
# Lab only — does not touch production serve ports.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LLAMA_CPP="${CLEF_CONVERT_LLAMA_CPP:-${ROOT}/vendor/llama-cpp-b11351}"
SRC="${CLEF_CONVERT_SRC:-/root/models/clef-flash-hf}"
OUT="${CLEF_CONVERT_OUT:-/root/models/clef-flash-gguf/clef-flash-ollama-q8_0.gguf}"
OUTTYPE="${CLEF_CONVERT_OUTTYPE:-q8_0}"

if [[ ! -d "${SRC}" ]]; then
  echo "error: missing HF tree ${SRC}" >&2
  echo "hint: hf download Cloudflare/clef-flash --local-dir ${SRC}" >&2
  exit 1
fi

python3 "${ROOT}/llama/clef/convert.py" --llama-cpp "${LLAMA_CPP}" --check-only "${SRC}"

mkdir -p "$(dirname "${OUT}")"
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
