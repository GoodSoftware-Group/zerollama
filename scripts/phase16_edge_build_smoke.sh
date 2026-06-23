#!/usr/bin/env bash
# Phase 16 edge compile marker smoke — no GPU; proves -tags edge build + CLI marker.
#
# Usage:
#   ./scripts/phase16_edge_build_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${P16_EDGE_BIN:-/tmp/zerollama-edge-smoke}"

echo "== Phase 16 edge tag unit tests =="
(
  cd "${ROOT}"
  go test -tags edge -count=1 ./envconfig/ -run 'EdgeBuildTag|ApplyServeBackendEnvEdgeBuild|GgmlRunnerLinked'
  go test -tags edge -count=1 ./runner/ -run ExecuteEdgeStub
  go test -tags edge -count=1 ./llm/ -run GgmlRunnerRequired
  go test -tags edge -count=1 ./server/ -run InferenceBackendPolicyEdgeBuildGgmlUnlinked
  go test -tags edge -count=1 ./server/ -run SchedSkipGgmlRunnerLoadEdgeBuild
)

echo ""
echo "== Phase 16 edge binary build =="
"${ROOT}/scripts/build_zerollama_edge.sh" "${OUT}"

echo ""
echo "== Phase 16 edge CLI marker =="
if ! "${OUT}" -v 2>&1 | tee /tmp/phase16-edge-version.out | grep -q 'edge build: true'; then
  echo "FAIL: expected 'edge build: true' from ${OUT} -v" >&2
  cat /tmp/phase16-edge-version.out >&2
  exit 1
fi

echo ""
echo "== Phase 16 edge runner stub =="
runner_out=$("${OUT}" runner --model /tmp/edge-smoke.gguf 2>&1) || true
if ! grep -q 'not included in edge builds' <<< "${runner_out}"; then
  echo "FAIL: edge binary should reject zerollama runner subprocess" >&2
  echo "${runner_out}" >&2
  exit 1
fi

echo "PASS: phase16_edge_build_smoke (${OUT})"
echo "Doc: docs/phase16-thin-edge.md"
