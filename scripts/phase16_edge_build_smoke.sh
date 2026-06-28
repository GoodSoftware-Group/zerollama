#!/usr/bin/env bash
# Phase 16 edge compile marker smoke — no GPU; proves -tags edge build + CLI marker.
#
# WHY: v1/v2 edge artifacts must compile, expose edge_build in CLI, and reject ggml runner
# subprocess entry — without a GPU or llama-server E2E (see phase16_edge_smoke.sh for that).
#
# Usage:
#   ./scripts/phase16_edge_build_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${P16_EDGE_BIN:-/tmp/zerollama-edge-smoke}"
VERSION_OUT="${P16_EDGE_VERSION_OUT:-/tmp/phase16-edge-version.out}"

echo "== Phase 16 edge tag unit tests =="
(
  cd "${ROOT}"
  go test -tags edge -count=1 ./envconfig/ -run 'EdgeBuildTag|ApplyServeBackendEnvEdgeBuild|GgmlRunnerLinked|RuntimeEmbed|RuntimeDarwin'
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
# WHY capture then grep: grep -q in a pipe under pipefail gets SIGPIPE (141) when it
# closes early after a match; zerollama -v may also emit warnings — do not pipe either.
version_out=$("${OUT}" -v 2>&1) || true
printf '%s\n' "${version_out}" >"${VERSION_OUT}"
if ! grep -q 'edge build: true' <<< "${version_out}"; then
  echo "FAIL: expected 'edge build: true' from ${OUT} -v" >&2
  cat "${VERSION_OUT}" >&2
  exit 1
fi
echo "ok: edge build marker (${VERSION_OUT})"

echo ""
echo "== Phase 16 edge runner stub =="
runner_out=$("${OUT}" runner --model /tmp/edge-smoke.gguf 2>&1) || true
if ! grep -q 'not included in edge builds' <<< "${runner_out}"; then
  echo "FAIL: edge binary should reject zerollama runner subprocess" >&2
  echo "${runner_out}" >&2
  exit 1
fi
echo "ok: runner subprocess stub rejected"

echo "PASS: phase16_edge_build_smoke (${OUT})"
echo "Doc: docs/phase16-thin-edge.md"
