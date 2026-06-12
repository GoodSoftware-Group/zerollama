#!/usr/bin/env bash
# Clone upstream Ollama beside zerollama for compare/contrast (no merge into zerollama).
#
# Usage:
#   ./scripts/clone_upstream_ollama.sh
#   OLLAMA_UPSTREAM_DIR=~/Sites/inference/ollama-upstream ./scripts/clone_upstream_ollama.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${OLLAMA_UPSTREAM_DIR:-${ROOT}/../ollama-upstream}"
URL="${OLLAMA_UPSTREAM_URL:-https://github.com/ollama/ollama.git}"

if [[ -d "${DEST}/.git" ]]; then
  echo ">>> upstream ollama already at ${DEST}" >&2
  git -C "${DEST}" fetch origin
  git -C "${DEST}" log -1 --oneline
  exit 0
fi

echo ">>> cloning ${URL} -> ${DEST}" >&2
git clone "${URL}" "${DEST}"
git -C "${DEST}" log -1 --oneline
echo ">>> compare: diff -ruN ${ROOT}/server ${DEST}/server | less" >&2
echo ">>> doc: ${ROOT}/docs/upstream-ollama-diff.md" >&2
echo ">>> build upstream: ./scripts/build_upstream_ollama_mac.sh" >&2
echo ">>> A/B serve: OLLAMA_HOST=127.0.0.1:11435 ./ollama serve" >&2
