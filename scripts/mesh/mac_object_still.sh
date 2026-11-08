#!/usr/bin/env bash
# Make an object-centric still on the M4 Max, then rembg → RGBA PNG for TRELLIS.2.
#
# Prefer an already-running imagegen serve. If none is up, pass --image to skip
# generation so the astra path is not blocked.
#
# Usage:
#   ./scripts/mesh/mac_object_still.sh --prompt "a red ceramic mug, plain white background" --out still.png
#   ./scripts/mesh/mac_object_still.sh --image existing.png --out still.png   # rembg only
set -euo pipefail

PROMPT=""
IN_IMAGE=""
OUT="still.png"
HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
MODEL="${MESH_IMAGE_MODEL:-x/z-image-turbo}"
REMBG_VENV="${REMBG_VENV:-$HOME/.zerollama/third_party/trellis/.venv-rembg-mac}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt) PROMPT="${2:-}"; shift 2 ;;
    --image) IN_IMAGE="${2:-}"; shift 2 ;;
    --out) OUT="${2:-}"; shift 2 ;;
    --host) HOST="${2:-}"; shift 2 ;;
    --model) MODEL="${2:-}"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 (--prompt TEXT | --image PATH) [--out still.png] [--host URL] [--model TAG]"
      exit 0
      ;;
    *) echo "unknown: $1" >&2; exit 1 ;;
  esac
done

OUT_PATH="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUT")"
mkdir -p "$(dirname "$OUT_PATH")"
RAW="${OUT_PATH%.png}.raw.png"

if [[ -n "$IN_IMAGE" ]]; then
  RAW="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$IN_IMAGE")"
  [[ -f "$RAW" ]] || { echo "error: --image not found: $RAW" >&2; exit 2; }
elif [[ -n "$PROMPT" ]]; then
  # Object-centric prompt hygiene: force plain background if the user forgot.
  case "$PROMPT" in
    *background*|*Background*) ;;
    *) PROMPT="$PROMPT, single object, centered, plain white background, product photo" ;;
  esac
  echo "imagegen host=$HOST model=$MODEL"
  if ! curl -fsS -m 2 "${HOST%/}/api/tags" >/dev/null 2>&1; then
    echo "error: no serve at $HOST — start imagegen, or pass --image PATH" >&2
    exit 2
  fi
  # OpenAI-compatible generations when available; fall back to /api/generate.
  if curl -fsS -m 2 -X POST "${HOST%/}/v1/images/generations" \
      -H 'Content-Type: application/json' \
      -d "$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"prompt":sys.argv[2],"n":1,"size":"1024x1024","response_format":"b64_json"}))' "$MODEL" "$PROMPT")" \
      -o /tmp/mesh_still_openai.json 2>/dev/null; then
    python3 - <<PY
import base64, json, pathlib, sys
p = pathlib.Path("/tmp/mesh_still_openai.json")
d = json.loads(p.read_text())
b64 = None
if "data" in d and d["data"]:
    b64 = d["data"][0].get("b64_json") or d["data"][0].get("b64")
if not b64:
    print("error: no b64 in openai response", d.keys(), file=sys.stderr)
    sys.exit(1)
pathlib.Path("$RAW").write_bytes(base64.b64decode(b64))
print("wrote $RAW")
PY
  else
    curl -fsS -X POST "${HOST%/}/api/generate" \
      -H 'Content-Type: application/json' \
      -d "$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"prompt":sys.argv[2],"stream":False}))' "$MODEL" "$PROMPT")" \
      -o /tmp/mesh_still_generate.json
    python3 - <<PY
import base64, json, pathlib, sys
d = json.loads(pathlib.Path("/tmp/mesh_still_generate.json").read_text())
# Ollama imagegen returns base64 under various keys depending on path.
b64 = d.get("image") or d.get("images", [None])[0]
if isinstance(b64, list):
    b64 = b64[0] if b64 else None
if not b64:
    print("error: no image in /api/generate response; keys=", list(d.keys()), file=sys.stderr)
    sys.exit(1)
# Sometimes prefixed data URL
if isinstance(b64, str) and "," in b64 and b64.startswith("data:"):
    b64 = b64.split(",", 1)[1]
pathlib.Path("$RAW").write_bytes(base64.b64decode(b64))
print("wrote $RAW")
PY
  fi
else
  echo "error: need --prompt or --image" >&2
  exit 2
fi

# rembg in a tiny Mac venv (not the CUDA trellis venv).
# U2NET_HOME: model cache — keep under TRELLIS_ROOT/lab so home sandbox is fine.
U2NET_HOME="${U2NET_HOME:-$(dirname "$OUT_PATH")/u2net}"
mkdir -p "$U2NET_HOME"
export U2NET_HOME
if [[ ! -x "$REMBG_VENV/bin/python" ]]; then
  python3 -m venv "$REMBG_VENV"
  "$REMBG_VENV/bin/pip" install -U pip
  "$REMBG_VENV/bin/pip" install rembg onnxruntime pillow
fi

U2NET_HOME="$U2NET_HOME" "$REMBG_VENV/bin/python" - <<PY
import os
from pathlib import Path
from rembg import remove
from PIL import Image
os.environ["U2NET_HOME"] = os.environ.get("U2NET_HOME", "")
raw = Path("$RAW")
out = Path("$OUT_PATH")
img = Image.open(raw).convert("RGBA")
cut = remove(img)
cut.save(out)
print(f"wrote {out} size={cut.size} u2net={os.environ.get('U2NET_HOME')}")
PY

echo "Still ready: $OUT_PATH"
echo "Copy to astra, e.g.: scp $OUT_PATH cozmic:/mnt/ssd2/zerollama/third_party/trellis/lab/"
