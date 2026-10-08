#!/usr/bin/env bash
# Smoke Clef multimodal /v1/decisions (text GGUF + mmproj) on a lab or prod host.
#
# Prerequisites:
#   - tip LLAMA_SERVER_BIN with clef score_fields + mtmd
#   - model tag with projector layer (e.g. clef-flash-vl from two FROM lines)
#   - host already serving (this script does not start serve)
#
# Usage:
#   OLLAMA_HOST=127.0.0.1:8080 ./scripts/phase/l6_clef_vl_decisions_smoke.sh
#   CLEF_VL_MODEL=clef-flash-vl CLEF_VL_IMAGE=/path/to.jpg ./scripts/phase/l6_clef_vl_decisions_smoke.sh
set -euo pipefail

HOST="${OLLAMA_HOST:-127.0.0.1:8080}"
MODEL="${CLEF_VL_MODEL:-clef-flash-vl}"
IMG="${CLEF_VL_IMAGE:-}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -z "${IMG}" ]]; then
  for candidate in \
    "${ROOT}/llama/llama.cpp/tools/mtmd/test-1.jpeg" \
    "${ROOT}/vendor/llama-cpp-b11351/tools/mtmd/test-1.jpeg"; do
    if [[ -f "${candidate}" ]]; then
      IMG="${candidate}"
      break
    fi
  done
fi
if [[ -z "${IMG}" || ! -f "${IMG}" ]]; then
  echo "error: set CLEF_VL_IMAGE to a JPEG/PNG" >&2
  exit 1
fi

BASE="${HOST}"
if [[ "${BASE}" != http* ]]; then
  BASE="http://${BASE}"
fi

python3 - <<PY
import base64, json, sys, urllib.request
from pathlib import Path

img = Path("${IMG}").read_bytes()
if len(img) > 200_000:
    try:
        from PIL import Image
        import io
        im = Image.open("${IMG}").convert("RGB")
        im.thumbnail((256, 256))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=70)
        img = buf.getvalue()
    except Exception:
        pass

req = {
    "model": "${MODEL}",
    "state": "Is there a receipt or invoice visible that would support a refund?",
    "images": [base64.b64encode(img).decode()],
    "keep_alive": "2m",
    "options": {"num_ctx": int("${CLEF_VL_NUM_CTX:-2048}"), "num_gpu": int("${CLEF_VL_NUM_GPU:-99}")},
    "questions": {
        "refund": {
            "type": "noul",
            "instructions": "Refund justified based on the image?",
        }
    },
}
body = json.dumps(req).encode()
r = urllib.request.Request(
    "${BASE}/v1/decisions",
    data=body,
    headers={"content-type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(r, timeout=600) as resp:
    d = json.loads(resp.read().decode())
if "error" in d:
    print("FAIL", d["error"][:800], file=sys.stderr)
    sys.exit(1)
ans = d.get("answers") or {}
assert "refund" in ans, d
print("PASS model=", d.get("model"), "answers=", json.dumps(ans)[:400], "usage=", d.get("usage"))
PY
