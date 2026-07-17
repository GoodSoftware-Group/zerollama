#!/usr/bin/env bash
# L2 CUDA A/B — L1 (stock cache) vs fork (QJL/Polar) profiles on unified llama-server.
#
# WHY: L2 exit gate needs measured tok/s + memory before enabling fork profiles by default.
# One binary (elizaOS/llama.cpp @ LLAMA_CPP_COMMIT); legs differ by ZEROLLAMA_LLAMA_FORK only.
#
# Prerequisite:
#   ../llama.cpp/build/bin/llama-server    (./scripts/build/build_llama_server.sh)
#
# Usage:
#   CUDA_LLAMA_MODEL=/path/to/model.gguf ./scripts/phase/l2_cuda_bench.sh
#   L2_BUILD_FORK=1 ./scripts/phase/l2_cuda_bench.sh
#
# Env:
#   CUDA_LLAMA_MODEL         — GGUF path (required; or LLAMA_MODEL)
#   L2_NUM_CTX               — bench context (default: 8192)
#   L2_NUM_PREDICT           — decode tokens per run (default: 128)
#   L2_BENCH_RUNS            — timed runs after warmup (default: 2)
#   L2_BUILD=1 / L2_BUILD_FORK=1 — build ../llama.cpp before bench
#   L2_CUDA_BENCH_OUT        — comparison JSON (default: /tmp/l2-cuda-bench.json)
#   L2_ALLOW_AUTO_CONFIG=1   — allow dual-GPU autoconfig (default: force single_gpu.yaml)
#   CUDA_VISIBLE_DEVICES     — default 0 for clean A/B on multi-GPU hosts
#   L2_FORK_CACHE_TYPE_K/V   — override fork profile cache types (e.g. tbq4_0 / tbq3_0)
#   STOCK_LLAMA_CPP_ROOT     — default vendor/sibling (legacy name; same tree as fork leg)
#   ELIZA_LLAMA_CPP_ROOT     — default same as stock (legacy alias)
#   L2_SKIP_STOCK=1          — skip stock leg
#   L2_SKIP_FORK=1           — skip fork leg
#   L2_SKIP_PREFILL=1        — skip prefill measurement
#   L2_HIGH_CTX_WARMUPS      — extra short decodes before timed runs at num_ctx>=65536
#   LINUX_RT_HEALTH_MAX      — sidecar startup timeout in seconds (default: 120)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ZEROLLAMA_PARENT="$(cd "${ROOT}/.." && pwd)"
# shellcheck source=scripts/runtime/linux_runtime_serve_lib.sh
# WHY: linux_runtime_serve_lib already sources runtime_uv_venv and runtime_smoke_lib.
source "${ROOT}/scripts/runtime/linux_runtime_serve_lib.sh"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "warn: l2_cuda_bench targets Linux CUDA; continuing anyway" >&2
fi

runtime_uv_venv

if [[ -n "${LLAMA_CPP_ROOT:-}" && -x "${LLAMA_CPP_ROOT}/build/bin/llama-server" ]]; then
  UNIFIED_ROOT="${LLAMA_CPP_ROOT}"
elif [[ -n "${LLAMA_SERVER_BIN:-}" && -x "${LLAMA_SERVER_BIN}" ]]; then
  UNIFIED_ROOT="$(cd "$(dirname "${LLAMA_SERVER_BIN}")/../.." && pwd)"
elif UNIFIED_ROOT="$(l1_vendor_llama_cpp_root "${ROOT}" 2>/dev/null)"; then
  :
else
  UNIFIED_ROOT="${ZEROLLAMA_PARENT}/llama.cpp"
fi
STOCK_ROOT="${STOCK_LLAMA_CPP_ROOT:-${UNIFIED_ROOT}}"
FORK_ROOT="${ELIZA_LLAMA_CPP_ROOT:-${UNIFIED_ROOT}}"
L2_OUT="${L2_CUDA_BENCH_OUT:-/tmp/l2-cuda-bench.json}"
L2_NUM_CTX="${L2_NUM_CTX:-8192}"
L2_NUM_PREDICT="${L2_NUM_PREDICT:-128}"
L2_BENCH_RUNS="${L2_BENCH_RUNS:-2}"

# Resolve model path: CUDA_LLAMA_MODEL or LLAMA_MODEL fallback.
LLAMA_MODEL="${CUDA_LLAMA_MODEL:-${LLAMA_MODEL:-}}"
if [[ -z "${LLAMA_MODEL:-}" ]]; then
  echo "Set CUDA_LLAMA_MODEL (or LLAMA_MODEL) to a small GGUF for CUDA bench" >&2
  exit 1
fi
export LLAMA_MODEL

if [[ "${L2_BUILD:-0}" == "1" || "${L2_BUILD_FORK:-0}" == "1" ]]; then
  LLAMA_CPP_ROOT="${UNIFIED_ROOT}" "${ROOT}/scripts/build/build_llama_server.sh"
fi

SERVER_BIN="${UNIFIED_ROOT}/build/bin/llama-server"
export LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-${SERVER_BIN}}"
export LLAMA_CPP_ROOT="${UNIFIED_ROOT}"
linux_runtime_export_llama_ld_path

if [[ "${L2_SKIP_STOCK:-0}" != "1" || "${L2_SKIP_FORK:-0}" != "1" ]]; then
  if [[ ! -x "${SERVER_BIN}" ]]; then
    echo "Missing ${SERVER_BIN}; run ./scripts/build/build_llama_server.sh (or point LLAMA_CPP_ROOT at a built vendor tree)" >&2
    exit 1
  fi
fi

# WHY default-on but overridable: L2 gate expects profile ON; L1 calibration passes
# ZEROLLAMA_GPU_PROFILE=0 for OFF baseline (see scripts/phase/l1_cuda_calibrate.sh).
export ZEROLLAMA_GPU_PROFILE="${ZEROLLAMA_GPU_PROFILE:-1}"
export ZEROLLAMA_RUNTIME_LLAMA_BACKEND=subprocess
# WHY single GPU for A/B: autoconfig on dual-4090 hosts picks tensor-parallel YAML
# and tanks decode tok/s (~6 vs expected 30+) while conflating VRAM across cards.
if [[ -z "${ZEROLLAMA_RUNTIME_CONFIG:-}" && -z "${L2_ALLOW_AUTO_CONFIG:-}" ]]; then
  export ZEROLLAMA_RUNTIME_CONFIG="${ROOT}/runtime/configs/single_gpu.yaml"
  unset ZEROLLAMA_AUTO_CONFIG
else
  unset ZEROLLAMA_RUNTIME_CONFIG
  export ZEROLLAMA_AUTO_CONFIG=1
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LINUX_RT_HEALTH_MAX="${LINUX_RT_HEALTH_MAX:-120}"

linux_runtime_urls
trap linux_runtime_sidecar_cleanup EXIT

_L2_LEG_JSON=()

_l2_run_leg() {
  local label="$1"
  local cpp_root="$2"
  local fork_mode="$3"   # off | on

  echo ""
  echo "== L2 leg: ${label} (${cpp_root}) fork=${fork_mode} =="

  export LLAMA_CPP_ROOT="${cpp_root}"
  export LLAMA_SERVER_BIN="${cpp_root}/build/bin/llama-server"
  # WHY .so: macOS uses .dylib; Linux shared objects use .so. The runtime
  # worker (libllama_ctypes.py) resolves the library path via LLAMA_CPP_LIB.
  export LLAMA_CPP_LIB="${cpp_root}/build/bin/libllama.so"

  case "${fork_mode}" in
    off|0|stock)
      export ZEROLLAMA_LLAMA_FORK=0
      unset LLAMA_SERVER_EXTRA_ARGS
      ;;
    on|1|auto|fork)
      # WHY explicit 1 (not unset): empty config/`serve.llama_fork: stock` on
      # dual_4090 autoconfig would force stock when env is unset.
      export ZEROLLAMA_LLAMA_FORK=1
      # Optional cache override (e.g. tbq4_0/tbq3_0 when qjl aborts on a GGUF).
      # Appended last via LLAMA_SERVER_EXTRA_ARGS — llama-server last-wins.
      if [[ -n "${L2_FORK_CACHE_TYPE_K:-}" || -n "${L2_FORK_CACHE_TYPE_V:-}" ]]; then
        local extra=()
        [[ -n "${L2_FORK_CACHE_TYPE_K:-}" ]] && extra+=(--cache-type-k "${L2_FORK_CACHE_TYPE_K}")
        [[ -n "${L2_FORK_CACHE_TYPE_V:-}" ]] && extra+=(--cache-type-v "${L2_FORK_CACHE_TYPE_V}")
        export LLAMA_SERVER_EXTRA_ARGS="${extra[*]}${LLAMA_SERVER_EXTRA_ARGS:+ ${LLAMA_SERVER_EXTRA_ARGS}}"
        # Align /health llama_fork_profile with the override (default FORK_PROFILE is vram/TBQ).
        local _ck="${L2_FORK_CACHE_TYPE_K:-}" _cv="${L2_FORK_CACHE_TYPE_V:-}"
        if [[ "${_ck}${_cv}" == *qjl* || "${_ck}${_cv}" == *polar* ]]; then
          export ZEROLLAMA_LLAMA_FORK_PROFILE=speed
        elif [[ "${_ck}${_cv}" == *tbq* ]]; then
          export ZEROLLAMA_LLAMA_FORK_PROFILE=vram
        fi
      fi
      ;;
    *) echo "unknown fork_mode: ${fork_mode}" >&2; return 1 ;;
  esac

  # WHY: with ZEROLLAMA_GPU_PROFILE_CTX=0 the profile omits -c; without an explicit
  # -c llama-server falls back toward n_ctx_train (often 128k+) and blows VRAM.
  # Long-ctx legs set PROFILE_CTX=0 so L2_NUM_CTX must become argv.
  local _ctx_env
  _ctx_env="$(echo "${ZEROLLAMA_GPU_PROFILE_CTX:-}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${_ctx_env}" =~ ^(0|false|no|off)$ ]]; then
    export LLAMA_SERVER_EXTRA_ARGS="-c ${L2_NUM_CTX}${LLAMA_SERVER_EXTRA_ARGS:+ ${LLAMA_SERVER_EXTRA_ARGS}}"
  fi

  linux_runtime_stop_sidecar_port
  # WHY pass config path: start_sidecar "" clears ZEROLLAMA_RUNTIME_CONFIG and
  # re-enables autoconfig (dual_4090 → llama_fork: stock), defeating L2 A/B.
  linux_runtime_start_sidecar "${LLAMA_MODEL}" "${ZEROLLAMA_RUNTIME_CONFIG:-}"

  local health_json
  health_json="$(runtime_fetch_health "${ZEROLLAMA_RUNTIME_URL}")"
  linux_runtime_resume_if_needed "${health_json}"

  # Gate: refuse to bench if fork env did not land in the live sidecar.
  if ! echo "${health_json}" | python3 -c "
import json, os, sys
h = json.load(sys.stdin)
lf = h.get('llama_fork') or {}
want = os.environ.get('ZEROLLAMA_LLAMA_FORK', '')
enabled = bool(lf.get('enabled'))
if want in ('0', 'false', 'off', 'stock') and enabled:
    print('L2 leg health: expected fork off, got', lf, file=sys.stderr)
    sys.exit(1)
if want in ('1', 'true', 'on', 'fork', 'eliza') and not enabled:
    print('L2 leg health: expected fork on, got', lf, file=sys.stderr)
    sys.exit(1)
print('llama_fork:', lf)
"; then
    return 1
  fi

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
    "List ten interesting facts about machine learning inference on NVIDIA CUDA. "
    "Number each fact.\n1."
)
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


def decode_tok_per_s(out: dict, elapsed_s: float, n_predict: int) -> float:
    """Prefer server eval timings; wall clock includes prefill + HTTP (last resort)."""
    eval_count = out.get("eval_count")
    eval_duration = out.get("eval_duration")  # ns
    if (
        isinstance(eval_count, int)
        and eval_count > 0
        and isinstance(eval_duration, (int, float))
        and eval_duration > 0
    ):
        return eval_count / (eval_duration / 1e9)
    if elapsed_s > 0:
        n = eval_count if isinstance(eval_count, int) and eval_count > 0 else n_predict
        return n / elapsed_s
    return 0.0


def nvidia_used_mib() -> list[int] | None:
    """Live device VRAM used (MiB) via nvidia-smi — preferred over heuristic estimates.

    When CUDA_VISIBLE_DEVICES is set, only those GPU indices are returned (same
    order as the env list). Otherwise all devices are sampled.
    """
    import subprocess

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    by_index: dict[int, int] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            by_index[int(parts[0])] = int(parts[1].split()[0])
        except ValueError:
            return None
    if not by_index:
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        vals: list[int] = []
        for tok in visible.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                idx = int(tok)
            except ValueError:
                continue
            if idx in by_index:
                vals.append(by_index[idx])
        return vals or None
    return [by_index[i] for i in sorted(by_index)]


health = http_json("GET", "/health")
estimate = http_json(
    "POST",
    "/internal/vram-estimate",
    {"gguf": gguf, "num_ctx": num_ctx},
)

# Profile argv for the report: rebuild RuntimeConfig with the same env as the live sidecar.
# WHY not gpu_profile.llama_server_args: /health exposes profile metadata, not merged argv.
from runtime.config import RuntimeConfig

cfg_path = Path("configs/single_gpu.yaml")
cfg = RuntimeConfig.from_file(cfg_path) if cfg_path.exists() else None
llama_args = cfg.llama_server_args() if cfg else []
gp_live = health.get("gpu_profile") or {}

# Warmup.
warmup_out, warmup_s = generate(decode_prompt, n_predict=min(16, num_predict))

# Extra warmup decodes for long-ctx legs (steady-state KV allocation).
high_ctx_warmups = 0
if num_ctx >= 65536:
    high_ctx_warmups = max(1, int(os.environ.get("L2_HIGH_CTX_WARMUPS", "2")))
    for _ in range(high_ctx_warmups):
        generate(decode_prompt, n_predict=min(8, num_predict))

decode_times: list[float] = []
decode_tps_runs: list[float] = []
eval_counts: list[int] = []
vram_samples_mib: list[list[int]] = []
for _ in range(bench_runs):
    out, elapsed = generate(decode_prompt, n_predict=num_predict)
    decode_times.append(elapsed)
    decode_tps_runs.append(decode_tok_per_s(out, elapsed, num_predict))
    if isinstance(out.get("eval_count"), int):
        eval_counts.append(out["eval_count"])
    sample = nvidia_used_mib()
    if sample is not None:
        vram_samples_mib.append(sample)

prefill_body, prefill_s = None, None
prefill_err = None
if os.environ.get("L2_SKIP_PREFILL", "0").strip().lower() not in ("1", "true", "yes"):
    try:
        prefill_body, prefill_s = generate(prefill_prompt, n_predict=prefill_n_predict)
    except Exception as e:
        prefill_err = str(e)

decode_mean = statistics.mean(decode_times)
decode_tps = statistics.mean(decode_tps_runs) if decode_tps_runs else 0.0
prefill_tps = (
    decode_tok_per_s(prefill_body, prefill_s, prefill_n_predict)
    if prefill_body is not None and prefill_s is not None
    else None
)

# Peak per-GPU and sum across GPUs during timed decode runs.
vram_peak_per_gpu_mib = None
vram_peak_sum_mib = None
if vram_samples_mib:
    n_gpu = max(len(s) for s in vram_samples_mib)
    peaks = [0] * n_gpu
    for sample in vram_samples_mib:
        for i, v in enumerate(sample):
            if v > peaks[i]:
                peaks[i] = v
    vram_peak_per_gpu_mib = peaks
    vram_peak_sum_mib = sum(peaks)

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
        (llama_args[i + 1] for i in range(len(llama_args) - 1, -1, -1) if llama_args[i] == "--cache-type-k"),
        None,
    ),
    "cache_type_v": next(
        (llama_args[i + 1] for i in range(len(llama_args) - 1, -1, -1) if llama_args[i] == "--cache-type-v"),
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
        "decode_tok_per_s_runs": [round(x, 2) for x in decode_tps_runs],
        "eval_count_runs": eval_counts,
        "prefill_prompt_words": _prefill_words,
        "prefill_n_predict": prefill_n_predict,
        "prefill_decode_wall_s": round(prefill_s, 3) if prefill_s is not None else None,
        "prefill_decode_tok_per_s": round(prefill_tps, 2) if prefill_tps is not None else None,
        "prefill_error": prefill_err,
        "nvidia_smi_peak_per_gpu_mib": vram_peak_per_gpu_mib,
        "nvidia_smi_peak_sum_mib": vram_peak_sum_mib,
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
  _l2_run_leg "stock" "${STOCK_ROOT}" "${L2_STOCK_FORK_MODE:-off}"
fi
if [[ "${L2_SKIP_FORK:-0}" != "1" ]]; then
  _l2_run_leg "fork" "${FORK_ROOT}" "${L2_FORK_FORK_MODE:-on}"
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
out_path = os.environ.get("L2_OUT", "/tmp/l2-cuda-bench.json")

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
    s_nv = stock["bench"].get("nvidia_smi_peak_sum_mib")
    f_nv = fork["bench"].get("nvidia_smi_peak_sum_mib")
    nv_delta_pct = None
    if s_nv and f_nv and s_nv > 0:
        nv_delta_pct = (f_nv - s_nv) / s_nv * 100.0
    # Prefer live nvidia-smi for fork_wins_vram when both legs sampled it.
    if s_nv is not None and f_nv is not None:
        fork_wins_vram = f_nv < s_nv
    else:
        fork_wins_vram = f_req < s_req if s_req and f_req else None
    comparison = {
        "stock_decode_tok_per_s": s_tps,
        "fork_decode_tok_per_s": f_tps,
        "decode_delta_pct": round(delta_pct, 2) if delta_pct is not None else None,
        "fork_wins_decode": f_tps > s_tps if s_tps and f_tps else None,
        "stock_vram_required_bytes": s_req,
        "fork_vram_required_bytes": f_req,
        "vram_delta_pct": round(vram_delta_pct, 2) if vram_delta_pct is not None else None,
        "stock_nvidia_smi_peak_sum_mib": s_nv,
        "fork_nvidia_smi_peak_sum_mib": f_nv,
        "nvidia_smi_delta_pct": round(nv_delta_pct, 2) if nv_delta_pct is not None else None,
        "fork_wins_vram": fork_wins_vram,
        "stock_cache": [stock.get("cache_type_k"), stock.get("cache_type_v")],
        "fork_cache": [fork.get("cache_type_k"), fork.get("cache_type_v")],
    }

report = {
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "platform": "linux-cuda",
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
    if comparison.get("stock_nvidia_smi_peak_sum_mib") is not None:
        print(
            f"nvidia-smi peak sum MiB: stock={comparison.get('stock_nvidia_smi_peak_sum_mib')} "
            f"fork={comparison.get('fork_nvidia_smi_peak_sum_mib')} "
            f"delta={comparison.get('nvidia_smi_delta_pct')}% "
            f"fork_wins_vram={comparison.get('fork_wins_vram')}"
        )
PY

linux_runtime_stop_sidecar_port
trap - EXIT

echo ""
echo "PASS: l2_cuda_bench (${L2_OUT})"
echo "Next: RUN_E2E_QWEN35=1 qwen35 smoke on each binary; raise L2_NUM_CTX for long-ctx gate."
echo "Doc: docs/gpu-profiles-l2.md"
