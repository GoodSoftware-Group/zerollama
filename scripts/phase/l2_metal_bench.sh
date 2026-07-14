#!/usr/bin/env bash
# L2 Metal A/B — L1 vs fork GPU profiles on unified llama-server (Darwin/CUDA subprocess path).
#
# WHY: L2 exit gate needs measured tok/s + memory on M-series before fork profiles ship default-on.
# One binary at ../llama.cpp; legs differ by ZEROLLAMA_LLAMA_FORK only.
#
# Prerequisite:
#   ../llama.cpp/build/bin/llama-server    (./scripts/build/build_llama_server.sh)
#
# Usage:
#   M3_LLAMA_MODEL=/path/to/model.gguf ./scripts/phase/l2_metal_bench.sh
#   L2_BUILD_FORK=1 ./scripts/phase/l2_metal_bench.sh
#
# Env:
#   M3_LLAMA_MODEL           — GGUF path (required if no local blob found)
#   L2_NUM_CTX               — bench context (default: 8192)
#   L2_NUM_PREDICT           — decode tokens per run (default: 128)
#   L2_BENCH_RUNS            — timed runs after warmup (default: 2)
#   L2_BUILD=1 / L2_BUILD_FORK=1 — build ../llama.cpp before bench
#   L2_METAL_BENCH_OUT         — comparison JSON (default: /tmp/l2-metal-bench.json)
#   apple-silicon-128g profile — sets runtime KV pool 8192×16 for 131072 ctx admission
#   STOCK_LLAMA_CPP_ROOT       — default ../llama.cpp
#   ELIZA_LLAMA_CPP_ROOT       — default ../llama.cpp (legacy alias)
#   L2_SKIP_STOCK=1 / L2_SKIP_FORK=1 — run one leg only (debug)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ZEROLLAMA_PARENT="$(cd "${ROOT}/.." && pwd)"
# shellcheck source=scripts/runtime/runtime_uv_venv.sh
source "${ROOT}/scripts/runtime/runtime_uv_venv.sh"
# shellcheck source=scripts/runtime/runtime_smoke_lib.sh
source "${ROOT}/scripts/runtime/runtime_smoke_lib.sh"
# shellcheck source=scripts/runtime/macos_runtime_serve_lib.sh
source "${ROOT}/scripts/runtime/macos_runtime_serve_lib.sh"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "warn: l2_metal_bench targets Darwin Metal; continuing anyway" >&2
fi

runtime_uv_venv

UNIFIED_ROOT="${LLAMA_CPP_ROOT:-${ZEROLLAMA_PARENT}/llama.cpp}"
STOCK_ROOT="${STOCK_LLAMA_CPP_ROOT:-${UNIFIED_ROOT}}"
FORK_ROOT="${ELIZA_LLAMA_CPP_ROOT:-${UNIFIED_ROOT}}"
L2_OUT="${L2_METAL_BENCH_OUT:-/tmp/l2-metal-bench.json}"
L2_NUM_CTX="${L2_NUM_CTX:-8192}"
L2_NUM_PREDICT="${L2_NUM_PREDICT:-128}"
L2_BENCH_RUNS="${L2_BENCH_RUNS:-2}"

if [[ "${L2_BUILD:-0}" == "1" || "${L2_BUILD_FORK:-0}" == "1" ]]; then
  LLAMA_CPP_ROOT="${UNIFIED_ROOT}" "${ROOT}/scripts/build/build_llama_server.sh"
fi

smoke_m3_resolve_signoff_model

SERVER_BIN="${UNIFIED_ROOT}/build/bin/llama-server"

if [[ "${L2_SKIP_STOCK:-0}" != "1" || "${L2_SKIP_FORK:-0}" != "1" ]]; then
  if [[ ! -x "${SERVER_BIN}" ]]; then
    echo "Missing ${SERVER_BIN}; run ./scripts/build/build_llama_server.sh" >&2
    exit 1
  fi
fi

export ZEROLLAMA_GPU_PROFILE=1
export ZEROLLAMA_RUNTIME_LLAMA_BACKEND=subprocess
unset ZEROLLAMA_RUNTIME_CONFIG
export ZEROLLAMA_AUTO_CONFIG=1
# Large GGUF + profile -c can exceed default 30s sidecar boot wait.
export MACOS_RT_HEALTH_MAX="${MACOS_RT_HEALTH_MAX:-120}"

macos_runtime_urls
trap macos_runtime_sidecar_cleanup EXIT

_L2_LEG_JSON=()

_l2_run_leg() {
  local label="$1"
  local cpp_root="$2"
  local fork_mode="$3"   # off | auto

  echo ""
  echo "== L2 leg: ${label} (${cpp_root}) fork=${fork_mode} =="

  export LLAMA_CPP_ROOT="${cpp_root}"
  export LLAMA_SERVER_BIN="${cpp_root}/build/bin/llama-server"
  export LLAMA_CPP_LIB="${cpp_root}/build/bin/libllama.dylib"
  export LLAMA_MODEL

  case "${fork_mode}" in
    off) export ZEROLLAMA_LLAMA_FORK=0 ;;
    auto) unset ZEROLLAMA_LLAMA_FORK ;;
    *) echo "unknown fork_mode: ${fork_mode}" >&2; return 1 ;;
  esac

  macos_runtime_stop_sidecar_port
  macos_runtime_start_sidecar "${LLAMA_MODEL}" "" 0

  local health_json
  health_json="$(runtime_fetch_health "${ZEROLLAMA_RUNTIME_URL}")"
  runtime_resume_if_needed "${health_json}"

  local leg_json
  leg_json="$(
    export RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL}"
    export L2_LEG_LABEL="${label}"
    export L2_LEG_CPP_ROOT="${cpp_root}"
    export L2_LEG_FORK_MODE="${fork_mode}"
    export L2_NUM_CTX L2_NUM_PREDICT L2_BENCH_RUNS LLAMA_MODEL
    cd "${ROOT}/runtime" && PYTHONPATH=. "${RUNTIME_UV_PYTHON}" <<'PY'
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

runtime_url = os.environ["RUNTIME_URL"].rstrip("/")
gguf = os.environ["LLAMA_MODEL"]
num_ctx = int(os.environ.get("L2_NUM_CTX", "8192"))
num_predict = int(os.environ.get("L2_NUM_PREDICT", "128"))
bench_runs = max(1, int(os.environ.get("L2_BENCH_RUNS", "2")))

decode_prompt = (
    "List ten interesting facts about machine learning inference on Apple Silicon. "
    "Number each fact.\n1."
)
# Keep prefill within ~half of num_ctx (rough word→token ratio).
_prefill_words = max(16, min(256, num_ctx // 12))
prefill_prompt = ("The quick brown fox jumps over the lazy dog. " * _prefill_words).strip()
prefill_n_predict = min(64, num_predict)


def http_json(method: str, path: str, body: dict | None = None, timeout: float = 600.0):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"{runtime_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def generate(prompt: str, *, n_predict: int) -> tuple[dict, float]:
    payload = {
        "model": "l2-bench",
        "prompt": prompt,
        "stream": False,
        "options": {
            "gguf": gguf,
            "num_ctx": num_ctx,
            "num_predict": n_predict,
        },
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
estimate = http_json(
    "POST",
    "/internal/vram-estimate",
    {"gguf": gguf, "num_ctx": num_ctx},
)

# Profile argv: prefer live gpu_profile.llama_server_args from /health when present;
# otherwise read apple_silicon.yaml.  WHY fallback: this is metadata for the report only.
gp_live = health.get("gpu_profile") or {}
llama_args = gp_live.get("llama_server_args") or []
if not llama_args:
    from runtime.config import RuntimeConfig
    cfg_path = Path("configs/apple_silicon.yaml")
    cfg = RuntimeConfig.from_file(cfg_path) if cfg_path.exists() else None
    llama_args = cfg.llama_server_args() if cfg else []
else:
    cfg = None

# Warmup load + short decode.
_, warmup_s = generate(decode_prompt, n_predict=min(16, num_predict))

# High-ctx legs: extra warmup decodes so timed runs measure steady-state tok/s,
# not first-touch KV allocation at full num_ctx.
high_ctx_warmups = 0
if num_ctx >= 65536:
    high_ctx_warmups = max(1, int(os.environ.get("L2_HIGH_CTX_WARMUPS", "2")))
    for _ in range(high_ctx_warmups):
        generate(decode_prompt, n_predict=min(8, num_predict))

decode_times: list[float] = []
for _ in range(bench_runs):
    _, elapsed = generate(decode_prompt, n_predict=num_predict)
    decode_times.append(elapsed)

prefill_body, prefill_s = None, None
prefill_err = None
if os.environ.get("L2_SKIP_PREFILL", "0").strip().lower() not in ("1", "true", "yes"):
    try:
        prefill_body, prefill_s = generate(prefill_prompt, n_predict=prefill_n_predict)
    except Exception as e:
        prefill_err = str(e)

decode_mean = statistics.mean(decode_times)
decode_tps = num_predict / decode_mean if decode_mean > 0 else 0.0
prefill_tps = (
    prefill_n_predict / prefill_s if prefill_s and prefill_s > 0 else None
)

gp = health.get("gpu_profile") or {}
lf = health.get("llama_fork") or {}
ve = (estimate.get("vram_estimate") or {}) if estimate else {}
vb = (estimate.get("vram_budget") or {}) if estimate else {}

leg = {
    "label": os.environ["L2_LEG_LABEL"],
    "llama_cpp_root": os.environ["L2_LEG_CPP_ROOT"],
    "llama_server_bin": str(cfg.llama_server_bin) if (cfg and cfg.llama_server_bin) else os.environ.get("LLAMA_SERVER_BIN"),
    "fork_mode": os.environ["L2_LEG_FORK_MODE"],
    "llama_fork": lf,
    "gpu_profile": gp,
    "llama_server_args": llama_args,
    "cache_type_k": next(
        (llama_args[i + 1] for i, a in enumerate(llama_args) if a == "--cache-type-k"),
        None,
    ),
    "cache_type_v": next(
        (llama_args[i + 1] for i, a in enumerate(llama_args) if a == "--cache-type-v"),
        None,
    ),
    "bench": {
        "gguf": gguf,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "warmup_wall_s": round(warmup_s, 3),
        "high_ctx_warmup_decodes": high_ctx_warmups,
        "decode_wall_s_mean": round(decode_mean, 3),
        "decode_wall_s_runs": [round(x, 3) for x in decode_times],
        "decode_tok_per_s": round(decode_tps, 2),
        "prefill_prompt_words": _prefill_words,
        "prefill_n_predict": prefill_n_predict,
        "prefill_decode_wall_s": round(prefill_s, 3) if prefill_s is not None else None,
        "prefill_decode_tok_per_s": round(prefill_tps, 2) if prefill_tps is not None else None,
        "prefill_error": prefill_err,
    },
    "vram_estimate": {
        "required_per_gpu_bytes": ve.get("required_per_gpu_bytes"),
        "estimate_factor_effective": ve.get("estimate_factor_effective"),
        "num_ctx": ve.get("num_ctx"),
    },
    "vram_budget": {
        "fits_with_margin": vb.get("fits_with_margin"),
        "suggested_max_num_ctx": vb.get("suggested_max_num_ctx"),
    },
    "admission_gpu_free_bytes": (health.get("admission") or {}).get("gpu_free_bytes"),
}
print(json.dumps(leg))
PY
  )"
  _L2_LEG_JSON+=("${leg_json}")
  echo "  decode: $(echo "${leg_json}" | python3 -c 'import json,sys; b=json.load(sys.stdin)["bench"]; print(b["decode_tok_per_s"], "tok/s")')"
}

if [[ "${L2_SKIP_STOCK:-0}" != "1" ]]; then
  _l2_run_leg "stock" "${STOCK_ROOT}" "off"
fi
if [[ "${L2_SKIP_FORK:-0}" != "1" ]]; then
  _l2_run_leg "fork" "${FORK_ROOT}" "auto"
fi

export L2_OUT L2_NUM_CTX L2_NUM_PREDICT
export L2_LEGS_JSON
L2_LEGS_JSON="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${_L2_LEG_JSON[@]}")"

python3 <<'PY'
import datetime
import json
import os

legs_raw = json.loads(os.environ["L2_LEGS_JSON"])
legs = [json.loads(x) for x in legs_raw if x.strip()]
out_path = os.environ.get("L2_OUT", "/tmp/l2-metal-bench.json")

comparison = {}
if len(legs) == 2:
    a, b = legs[0], legs[1]
    stock = a if a["label"] == "stock" else b
    fork = b if a["label"] == "stock" else a
    s_tps = stock["bench"]["decode_tok_per_s"]
    f_tps = fork["bench"]["decode_tok_per_s"]
    delta_pct = ((f_tps - s_tps) / s_tps * 100.0) if s_tps > 0 else None
    s_req = stock["vram_estimate"].get("required_per_gpu_bytes")
    f_req = fork["vram_estimate"].get("required_per_gpu_bytes")
    vram_delta_pct = None
    if s_req and f_req and s_req > 0:
        vram_delta_pct = (f_req - s_req) / s_req * 100.0
    comparison = {
        "stock_decode_tok_per_s": s_tps,
        "fork_decode_tok_per_s": f_tps,
        "decode_delta_pct": round(delta_pct, 2) if delta_pct is not None else None,
        "fork_wins_decode": f_tps > s_tps if s_tps and f_tps else None,
        "stock_vram_required_bytes": s_req,
        "fork_vram_required_bytes": f_req,
        "vram_delta_pct": round(vram_delta_pct, 2) if vram_delta_pct is not None else None,
        "fork_wins_vram": f_req < s_req if s_req and f_req else None,
        "stock_cache": [stock.get("cache_type_k"), stock.get("cache_type_v")],
        "fork_cache": [fork.get("cache_type_k"), fork.get("cache_type_v")],
    }

report = {
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "platform": "darwin",
    "bench_params": {
        "num_ctx": int(os.environ.get("L2_NUM_CTX", "8192")),
        "num_predict": int(os.environ.get("L2_NUM_PREDICT", "128")),
    },
    "legs": {leg["label"]: leg for leg in legs},
    "comparison": comparison,
    "gate_notes": (
        "Automated leg covers decode tok/s + vram estimate at fixed num_ctx. "
        "L2 full gate also needs max stable ctx + qwen35 compat on both binaries."
    ),
}
text = json.dumps(report, indent=2)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(text + "\n")
print(f"wrote {out_path}")
if comparison:
    print(
        f"decode: stock={comparison.get('stock_decode_tok_per_s')} "
        f"fork={comparison.get('fork_decode_tok_per_s')} "
        f"delta={comparison.get('decode_delta_pct')}% "
        f"fork_wins={comparison.get('fork_wins_decode')}"
    )
PY

macos_runtime_stop_sidecar_port
trap - EXIT

echo ""
echo "PASS: l2_metal_bench (${L2_OUT})"
echo "Next: RUN_E2E_QWEN35=1 qwen35 smoke on each binary; raise L2_NUM_CTX for long-ctx gate."
echo "Doc: docs/gpu-profiles-l2.md"
