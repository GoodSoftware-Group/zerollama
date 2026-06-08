#!/usr/bin/env bash
# Phase 13 calibration snapshot for a GPU host (5080-class). No new loads beyond /health.
#
# WHY: portable JSON for before/after tuning; includes vram_autotune.persist for gpu_snapshot.
# WHY ZEROLLAMA_REPO_ROOT: recommend block imports runtime.gpu_snapshot without manual PYTHONPATH.
#
#   export LLAMA_MODEL=/path/to/model.gguf
#   ZEROLLAMA_RUNTIME_URL=http://127.0.0.1:8081 ./scripts/gpu_phase13_snapshot.sh
#   ./scripts/gpu_phase13_snapshot.sh --gguf /path/to/model.gguf --num-ctx 8192
#
# Writes JSON to stdout; optional file via GPU_PHASE13_SNAPSHOT_OUT=path
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ZEROLLAMA_REPO_ROOT="${ZEROLLAMA_REPO_ROOT:-$ROOT}"
RUNTIME_URL="${ZEROLLAMA_RUNTIME_URL:-http://127.0.0.1:8081}"
GGUF="${LLAMA_MODEL:-}"
NUM_CTX="${RUN_E2E_NUM_CTX:-8192}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gguf) GGUF="$2"; shift 2 ;;
    --num-ctx) NUM_CTX="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,8p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

health_json=$(curl -sf -m 30 "${RUNTIME_URL%/}/health")
estimate_json=""
if [[ -n "$GGUF" ]]; then
  _estimate_body=$(python3 -c 'import json,sys; print(json.dumps({"gguf":sys.argv[1],"num_ctx":int(sys.argv[2])}))' "$GGUF" "$NUM_CTX")
  estimate_json=$(curl -sf -m 120 -X POST "${RUNTIME_URL%/}/internal/vram-estimate" \
    -H 'Content-Type: application/json' \
    -d "${_estimate_body}")
fi

export HEALTH_JSON="$health_json" ESTIMATE_JSON="$estimate_json" SNAPSHOT_GGUF="$GGUF" SNAPSHOT_NUM_CTX="$NUM_CTX"
python3 <<'PY'
import json, os, sys, datetime

def pick(d, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

health = json.loads(os.environ["HEALTH_JSON"])
est = json.loads(os.environ["ESTIMATE_JSON"]) if os.environ.get("ESTIMATE_JSON") else {}
vb = health.get("vram_budget") or {}
vc = health.get("vram_calibration") or {}
va = health.get("vram_autotune") or {}
ad = health.get("admission") or {}

out = {
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "gguf": os.environ.get("SNAPSHOT_GGUF") or None,
    "num_ctx_probe": int(os.environ.get("SNAPSHOT_NUM_CTX", "0") or 0),
    "inference_state": health.get("inference_state"),
    "llama_server": health.get("llama_server"),
    "llama_backend": health.get("llama_backend"),
    "llama_backend_source": health.get("llama_backend_source"),
    "autoconfig": health.get("autoconfig"),
    "vram_budget": {
        "admission_fits": vb.get("admission_fits"),
        "fits_with_margin": vb.get("fits_with_margin"),
        "suggested_max_num_ctx": vb.get("suggested_max_num_ctx"),
    },
    "vram_calibration": {
        "model": vc.get("model"),
        "suggested_estimate_factor": vc.get("suggested_estimate_factor"),
        "observed_bytes": vc.get("observed_bytes"),
        "age_s": vc.get("age_s"),
    },
    "vram_autotune": {
        "enabled": va.get("enabled"),
        "effective_factor": va.get("effective_factor"),
        "session_model": va.get("session_model"),
        "persist": va.get("persist"),
    },
    "admission": {
        "gpu_free_bytes": ad.get("gpu_free_bytes"),
        "vram_min_free_configured": ad.get("vram_min_free_configured"),
        "vram_training_reserve_configured": ad.get("vram_training_reserve_configured"),
    },
}
he = health.get("vram_estimate") or {}
ve_out: dict = {}
if he.get("estimate_factor_source"):
    ve_out["estimate_factor_source"] = he.get("estimate_factor_source")
if he.get("estimate_factor_effective") is not None:
    ve_out["estimate_factor_effective"] = he.get("estimate_factor_effective")
if est:
    ve = est.get("vram_estimate") or {}
    eb = est.get("vram_budget") or {}
    if ve.get("required_per_gpu_bytes") is not None:
        ve_out["required_per_gpu_bytes"] = ve.get("required_per_gpu_bytes")
    if ve.get("estimate_factor_effective") is not None:
        ve_out["estimate_factor_effective"] = ve.get("estimate_factor_effective")
    if ve.get("estimate_factor_source"):
        ve_out["estimate_factor_source"] = ve.get("estimate_factor_source")
    if ve.get("num_ctx") is not None:
        ve_out["num_ctx"] = ve.get("num_ctx")
    out["vram_estimate_budget"] = {
        "fits_with_margin": eb.get("fits_with_margin"),
        "suggested_max_num_ctx": eb.get("suggested_max_num_ctx"),
    }
if ve_out:
    out["vram_estimate"] = ve_out
lcp = health.get("llama_cpp")
if lcp:
    out["llama_cpp"] = lcp
if est and "vram_estimate_budget" not in out:
    eb = est.get("vram_budget") or {}
    out["vram_estimate_budget"] = {
        "fits_with_margin": eb.get("fits_with_margin"),
        "suggested_max_num_ctx": eb.get("suggested_max_num_ctx"),
    }

text = json.dumps(out, indent=2)
out_path = os.environ.get("GPU_PHASE13_SNAPSHOT_OUT", "").strip()
if out_path:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"wrote {out_path}")
else:
    print(text)

if os.environ.get("GPU_SNAPSHOT_RECOMMEND", "1").strip().lower() not in ("0", "false", "no"):
    root = os.environ.get("ZEROLLAMA_REPO_ROOT", "").strip()
    if root:
        sys.path.insert(0, os.path.join(root, "runtime"))
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(here, "..", "runtime"))
    try:
        from runtime.gpu_snapshot import format_snapshot_recommendations
        print()
        print(format_snapshot_recommendations(out))
    except Exception as e:
        print(f"# snapshot recommend skipped: {e}", file=sys.stderr)
PY
