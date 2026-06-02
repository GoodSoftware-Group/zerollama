#!/usr/bin/env bash
# Query runtime VRAM estimate + budget for a GGUF (Phase 13). No model load.
#
#   ./scripts/runtime_vram_estimate.sh /path/to/model.gguf
#   LLAMA_MODEL=/path/to/model.gguf ./scripts/runtime_vram_estimate.sh
#   ./scripts/runtime_vram_estimate.sh model.gguf --num-ctx 8192
#
# Requires zerollama serve (embedded or sidecar) on loopback :8081.
# Why loopback-only: same security model as /internal/vram-estimate (no remote budget probe).
set -euo pipefail

RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"
GGUF="${1:-${LLAMA_MODEL:-}}"
NUM_CTX="${VRAM_ESTIMATE_NUM_CTX:-}"

shift 2>/dev/null || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-ctx)
      NUM_CTX="${2:?--num-ctx requires a value}"
      shift 2
      ;;
    --url)
      RUNTIME_URL="${2:?--url requires a value}"
      shift 2
      ;;
    -h|--help)
      sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$GGUF" ]]; then
  echo "usage: $0 <gguf-path> [--num-ctx N]" >&2
  echo "  or set LLAMA_MODEL" >&2
  exit 2
fi
if [[ ! -f "$GGUF" ]]; then
  echo "gguf not found: $GGUF" >&2
  exit 1
fi

export NUM_CTX
payload=$(python3 -c '
import json, os, sys
gguf = sys.argv[1]
num_ctx = os.environ.get("NUM_CTX", "").strip()
body = {"gguf": gguf, "options": {}}
if num_ctx:
    body["num_ctx"] = int(num_ctx)
    body["options"]["num_ctx"] = int(num_ctx)
print(json.dumps(body))
' "$GGUF")

tmp=$(mktemp)
code=$(curl -sS -o "$tmp" -w "%{http_code}" -X POST \
  -H 'Content-Type: application/json' \
  -d "$payload" \
  "${RUNTIME_URL}/internal/vram-estimate")
if [[ "$code" != "200" ]]; then
  echo "HTTP ${code} POST /internal/vram-estimate:" >&2
  cat "$tmp" >&2
  rm -f "$tmp"
  exit 1
fi

python3 -c '
import json, sys

def fmt_b(n):
    if n is None:
        return "?"
    n = int(n)
    for u, s in ((1 << 30, "GiB"), (1 << 20, "MiB"), (1 << 10, "KiB")):
        if n >= u:
            return f"{n / u:.2f} {s}"
    return f"{n} B"

body = json.load(open(sys.argv[1]))
est = body.get("vram_estimate") or {}
bud = body.get("vram_budget") or {}
print("gguf:", est.get("gguf"))
print("num_ctx:", est.get("num_ctx"))
req = est.get("required_per_gpu_bytes")
if req is not None:
    print("required_per_gpu:", fmt_b(req))
if est.get("estimate_factor_effective") is not None:
    print("estimate_factor_effective:", est.get("estimate_factor_effective"))
kv = est.get("kv_cache_bytes")
if kv is not None:
    print("kv_cache:", fmt_b(kv))
if bud.get("free_bottleneck") is not None:
    print("free_bottleneck:", fmt_b(bud.get("free_bottleneck")))
if bud.get("fits_with_margin") is not None:
    print("fits_with_margin:", bud.get("fits_with_margin"))
if bud.get("admission_fits") is not None:
    print("admission_fits:", bud.get("admission_fits"))
suggest = bud.get("suggested_max_num_ctx")
if suggest is not None:
    print("suggested_max_num_ctx:", suggest)
    if bud.get("num_ctx_over_budget"):
        print("num_ctx_over_budget: true")
host = bud.get("host_ram") or {}
if host:
    print("host_ram.fits:", host.get("fits"))
' "$tmp"
rm -f "$tmp"
