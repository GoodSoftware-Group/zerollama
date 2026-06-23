#!/usr/bin/env bash
# L3 prefix cache × speculative decode policy smoke.
#
# WHY: draft-based spec (eagle3/mtp/dflash) must disable RAM cache_prompt and disk
# slot persistence; ngram/none must keep prefix cache enabled. Generate with
# prompt_cache_key must still succeed when cache is policy-disabled.
#
# Usage:
#   M3_LLAMA_MODEL=/path/model.gguf ./scripts/l3_spec_cache_smoke.sh
#   L3_SPEC_METHOD=ngram ./scripts/l3_spec_cache_smoke.sh
#   L3_SPEC_METHOD=eagle3 LLAMA_DRAFT_MODEL=/path/draft.gguf ./scripts/l3_spec_cache_smoke.sh
#
# Env:
#   CUDA_LLAMA_MODEL / M3_LLAMA_MODEL — GGUF path
#   L3_SPEC_METHOD        — none | ngram | eagle3 | mtp | dflash | draft-simple (default ngram)
#   LLAMA_DRAFT_MODEL     — required for draft methods
#   L3_CACHE_KEY            — default l3-spec-smoke-1
#   L3_NUM_CTX              — default 8192
#   L3_NUM_PREDICT          — default 16
#   L3_OUT                  — JSON report (default /tmp/l3-spec-cache-smoke.json)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/runtime_uv_venv.sh
source "${ROOT}/scripts/runtime_uv_venv.sh"
# shellcheck source=scripts/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime_smoke_lib.sh"

_L3_LINUX=0
if [[ "$(uname -s)" == "Linux" ]]; then
  _L3_LINUX=1
  # shellcheck source=scripts/linux_runtime_serve_lib.sh
  source "${ROOT}/scripts/linux_runtime_serve_lib.sh"
else
  # shellcheck source=scripts/macos_runtime_serve_lib.sh
  source "${ROOT}/scripts/macos_runtime_serve_lib.sh"
fi

runtime_uv_venv

if [[ -n "${CUDA_LLAMA_MODEL:-}" ]]; then
  export M3_LLAMA_MODEL="${CUDA_LLAMA_MODEL}"
fi
smoke_m3_resolve_signoff_model

L3_SPEC_METHOD="${L3_SPEC_METHOD:-ngram}"
export ZEROLLAMA_SPEC_METHOD="${L3_SPEC_METHOD}"

_draft_methods=(eagle3 mtp dflash draft draft-simple)
_needs_draft=0
for _m in "${_draft_methods[@]}"; do
  if [[ "${L3_SPEC_METHOD}" == "${_m}" ]]; then
    _needs_draft=1
    break
  fi
done
if [[ "${_needs_draft}" == "1" ]]; then
  if [[ -z "${LLAMA_DRAFT_MODEL:-}" || ! -f "${LLAMA_DRAFT_MODEL}" ]]; then
    echo "skip: L3_SPEC_METHOD=${L3_SPEC_METHOD} requires LLAMA_DRAFT_MODEL (existing GGUF)" >&2
    exit 0
  fi
  export LLAMA_DRAFT_MODEL
fi

export ZEROLLAMA_GPU_PROFILE=1
export ZEROLLAMA_GPU_PROFILE_CTX=0
export ZEROLLAMA_RUNTIME_LLAMA_BACKEND=subprocess
unset ZEROLLAMA_RUNTIME_CONFIG
export ZEROLLAMA_AUTO_CONFIG=1
export ZEROLLAMA_LLAMA_FORK=0

export LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-$(cd "${ROOT}/.." && pwd)/llama.cpp}"
export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${LLAMA_CPP_ROOT}/build/bin/llama-server}"
if [[ "${_L3_LINUX}" == "1" ]]; then
  export LLAMA_CPP_LIB="${LLAMA_CPP_LIB:-${LLAMA_CPP_ROOT}/build/bin/libllama.so}"
  export LINUX_RT_HEALTH_MAX="${LINUX_RT_HEALTH_MAX:-180}"
  export ZEROLLAMA_GPU_PROFILE_CTX=1
else
  export LLAMA_CPP_LIB="${LLAMA_CPP_LIB:-${LLAMA_CPP_ROOT}/build/bin/libllama.dylib}"
  export MACOS_RT_HEALTH_MAX="${MACOS_RT_HEALTH_MAX:-90}"
fi

L3_OUT="${L3_OUT:-/tmp/l3-spec-cache-smoke.json}"
L3_CACHE_KEY="${L3_CACHE_KEY:-l3-spec-smoke-1}"
L3_NUM_CTX="${L3_NUM_CTX:-8192}"
L3_NUM_PREDICT="${L3_NUM_PREDICT:-16}"

if [[ "${_L3_LINUX}" == "1" ]]; then
  linux_runtime_urls
  trap linux_runtime_sidecar_cleanup EXIT
  linux_runtime_stop_sidecar_port
  linux_runtime_start_sidecar "" ""
else
  macos_runtime_urls
  trap macos_runtime_sidecar_cleanup EXIT
  macos_runtime_stop_sidecar_port
  macos_runtime_start_sidecar "" "" 0
fi

health_json="$(runtime_fetch_health "${ZEROLLAMA_RUNTIME_URL}")"
runtime_resume_if_needed "${health_json}"

export RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL}"
export LLAMA_MODEL L3_CACHE_KEY L3_NUM_CTX L3_NUM_PREDICT L3_OUT L3_SPEC_METHOD
(cd "${ROOT}/runtime" && PYTHONPATH=. "${RUNTIME_UV_PYTHON}" <<'PY'
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

url = os.environ["RUNTIME_URL"].rstrip("/")
gguf = os.environ["LLAMA_MODEL"]
cache_key = os.environ["L3_CACHE_KEY"]
num_ctx = int(os.environ.get("L3_NUM_CTX", "8192"))
n_predict = int(os.environ.get("L3_NUM_PREDICT", "16"))
out_path = Path(os.environ.get("L3_OUT", "/tmp/l3-spec-cache-smoke.json"))
spec_method = os.environ.get("L3_SPEC_METHOD", "ngram").strip().lower()

DRAFT_METHODS = frozenset(
    {"eagle3", "mtp", "dflash", "draft", "draft-simple", "draft_simple"}
)
expect_draft = spec_method in DRAFT_METHODS or spec_method.startswith("draft")


def http_json(method: str, path: str, body: dict | None = None, timeout: float = 600.0):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"{url}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def generate(prompt: str, *, cache_key: str | None) -> tuple[dict, float]:
    opts: dict = {"gguf": gguf, "num_ctx": num_ctx, "num_predict": n_predict}
    if cache_key:
        opts["prompt_cache_key"] = cache_key
    payload = {
        "model": "l3-spec-smoke",
        "prompt": prompt,
        "stream": False,
        "options": opts,
    }
    t0 = time.perf_counter()
    try:
        out = http_json("POST", "/api/generate", payload)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace") if e.fp else ""
        raise RuntimeError(f"generate HTTP {e.code}: {body[:500]}") from e
    elapsed = time.perf_counter() - t0
    if not out.get("done"):
        raise RuntimeError(f"generate incomplete: {out!r}")
    return out, elapsed


health = http_json("GET", "/health")
lc = health.get("llama_cache") or {}
policy = lc.get("policy") or {}
runtime_spec = (health.get("speculative_method") or "").strip().lower()

if lc.get("enabled") is False:
    raise SystemExit("ZEROLLAMA_LLAMA_CACHE disabled — enable for L3 spec smoke")

errors: list[str] = []
if expect_draft:
    if not policy.get("speculative_draft"):
        errors.append(
            f"expected policy.speculative_draft=true for {spec_method}, got {policy!r}"
        )
    if policy.get("allow_cache_prompt"):
        errors.append("draft spec must set allow_cache_prompt=false")
    if policy.get("allow_disk_persist"):
        errors.append("draft spec must set allow_disk_persist=false")
    notes = policy.get("notes") or []
    if "cache_prompt_disabled_draft_speculative" not in notes:
        errors.append("expected cache_prompt_disabled_draft_speculative in policy.notes")
else:
    if policy.get("speculative_draft"):
        errors.append(
            f"expected policy.speculative_draft=false for {spec_method}, got {policy!r}"
        )
    if not policy.get("allow_cache_prompt"):
        errors.append(f"{spec_method} should allow allow_cache_prompt=true")

if errors:
    for e in errors:
        print(f"FAIL: {e}", file=sys.stderr)
    raise SystemExit(1)

prompt = "User: Reply with one word.\nAssistant:"
body, wall_s = generate(prompt, cache_key=cache_key)
preview = (body.get("response") or "").strip()
if not preview:
    raise SystemExit("generate returned empty response")

report: dict = {
    "gguf": gguf,
    "spec_method": spec_method,
    "runtime_speculative_method": runtime_spec,
    "cache_key": cache_key,
    "num_ctx": num_ctx,
    "n_predict": n_predict,
    "llama_cache": lc,
    "policy": policy,
    "expect_draft_policy": expect_draft,
    "generate_wall_s": round(wall_s, 3),
    "response_preview": preview[:80],
}

if not expect_draft:
    _, wall2 = generate(prompt, cache_key=cache_key)
    report["second_generate_wall_s"] = round(wall2, 3)

out_path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
print(f"wrote {out_path}")
PY
)

if [[ "${_L3_LINUX}" == "1" ]]; then
  linux_runtime_stop_sidecar_port
else
  macos_runtime_stop_sidecar_port
fi
trap - EXIT

echo ""
echo "PASS: l3_spec_cache_smoke (${L3_OUT}, L3_SPEC_METHOD=${L3_SPEC_METHOD})"
echo "Doc: docs/gpu-profiles-l3.md runtime/docs/SPECULATIVE.md"
