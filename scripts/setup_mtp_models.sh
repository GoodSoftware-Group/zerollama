#!/usr/bin/env bash
# Enable speculative decoding for local models (MTP, n-gram, Eliza DFlash).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! curl -sf "${OLLAMA_HOST:-http://127.0.0.1:11434}/api/tags" >/dev/null 2>&1; then
  echo "zerollama server is not running — start it first (OLLAMA_HOST=${OLLAMA_HOST:-http://127.0.0.1:11434}):" >&2
  echo "  ./zerollama serve   # or ~/bin/serve.sh on CT 1564 (:8080)" >&2
  exit 1
fi

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

echo "==> Creating qwen3.6-mtp (embedded MTP, draft_num_predict=4)"
cat >"$TMP" <<'EOF'
FROM qwen3.6:latest
PARAMETER draft_num_predict 4
EOF
./zerollama create qwen3.6-mtp -f "$TMP"
echo ""

echo "==> Creating eliza n-gram variants (no extra weights)"
for tag in eliza-1-27b-256k-ngram eliza-1-2b-ngram; do
  ./zerollama create "$tag" -f "$ROOT/modelfiles/${tag}.modelfile"
done
echo ""

ELIZA_CACHE="${ELIZA_DFLASH_CACHE:-$HOME/.cache/zerollama/eliza-1}"
# HF layout (Aug 2026): bundles/e2b|e4b/… (legacy bundles/2b|27b-256k paths removed upstream).
DFLASH_2B="$ELIZA_CACHE/bundles/e2b/dflash/drafter-e2b.gguf"
DFLASH_27B="$ELIZA_CACHE/bundles/e4b/dflash/drafter-e4b.gguf"
# Fallback if operator still has legacy filenames.
[[ -f "$DFLASH_2B" ]] || DFLASH_2B="$ELIZA_CACHE/bundles/2b/dflash/drafter-2b.gguf"
[[ -f "$DFLASH_27B" ]] || DFLASH_27B="$ELIZA_CACHE/bundles/27b-256k/dflash/drafter-27b-256k.gguf"

if [[ -f "$DFLASH_2B" && -f "$DFLASH_27B" ]]; then
  echo "==> Creating eliza DFlash variants (draft-eagle3 from elizaos/eliza-1)"
  sed "s|DRAFT .*|DRAFT ${DFLASH_2B}|" "$ROOT/modelfiles/eliza-1-2b-dflash.modelfile" >"$TMP"
  ./zerollama create eliza-1-2b-dflash -f "$TMP"
  sed "s|DRAFT .*|DRAFT ${DFLASH_27B}|" "$ROOT/modelfiles/eliza-1-27b-256k-dflash.modelfile" >"$TMP"
  ./zerollama create eliza-1-27b-256k-dflash -f "$TMP"
else
  echo "Eliza DFlash drafters not cached. Download once:"
  echo "  pip3 install huggingface_hub"
  echo "  python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('elizaos/eliza-1', allow_patterns=['bundles/e2b/dflash/drafter-*.gguf','bundles/e4b/dflash/drafter-*.gguf'], local_dir='$ELIZA_CACHE')\""
  echo "Then re-run: $0"
fi

echo ""
echo "Use with llama-server on Mac (spec models auto-route when binary is built):"
echo "  ./scripts/build_ollama_llama_server_darwin.sh   # once, if not already on disk"
echo "  ./zerollama serve"
echo ""
echo "Or route all GGUF through llama-server:"
echo "  ./zerollama serve --llama-server-backend"
echo ""
echo "Tags:"
echo "  qwen3.6-mtp              — embedded MTP"
echo "  eliza-1-*-ngram          — n-gram (no draft file)"
echo "  eliza-1-*-dflash         — Eliza DFlash drafter (spec_type dflash; needs unified llama-server @ LLAMA_CPP_COMMIT)"
