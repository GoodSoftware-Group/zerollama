#!/usr/bin/env bash
# Fetch single-header deps for gliner-server (httplib + nlohmann/json).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TP="${ROOT}/gliner/server/third_party"
mkdir -p "${TP}/nlohmann"

HTTPLIB_URL="${HTTPLIB_URL:-https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.18.3/httplib.h}"
JSON_URL="${JSON_URL:-https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp}"

if [[ ! -f "${TP}/httplib.h" ]]; then
  echo ">>> fetch httplib.h" >&2
  curl -fsSL "${HTTPLIB_URL}" -o "${TP}/httplib.h"
fi
if [[ ! -f "${TP}/nlohmann/json.hpp" ]]; then
  echo ">>> fetch nlohmann/json.hpp" >&2
  curl -fsSL "${JSON_URL}" -o "${TP}/nlohmann/json.hpp"
fi
echo "${TP}"
