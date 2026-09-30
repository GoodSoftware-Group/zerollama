#!/usr/bin/env bash
# DG4a frozen-logit / ranking oracle: same seed+steps → same gather ranking.
# Lab only. Free heavy GPU jobs first. Never binds :11434/:8081.
#
# WHY ranking (not bit-exact floats): CUDA denoise may not be bitwise stable;
# OpenJev product needs stable argmax / relative order under a fixed seed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${DIFFUSION_GGUF:-/root/models/diffusiongemma/diffusiongemma-26B-A4B-it-Q4_K_M.gguf}"
NGL="${DIFFUSION_NGL:-8}"
CTX="${DIFFUSION_CTX:-2048}"
PORT="${DIFFUSION_PORT:-18093}"
HOST="${DIFFUSION_HOST:-127.0.0.1}"
STEPS="${DIFFUSION_STEPS:-8}"
SEED="${DIFFUSION_SEED:-42}"

export LD_LIBRARY_PATH="/root/nvidia-host:${LD_LIBRARY_PATH:-}"

if [[ -z "${ZEROLLAMA_DIFFUSION_SERVER_BIN:-}" ]]; then
  CAND="${ROOT}/vendor/llama-cpp-diffusion-dd0cf0445/build/bin/llama-diffusion-gemma-server"
  [[ -x "${CAND}" ]] || { echo "error: build llama-diffusion-gemma-server first" >&2; exit 1; }
  ZEROLLAMA_DIFFUSION_SERVER_BIN="${CAND}"
fi
[[ -f "${MODEL}" ]] || { echo "error: missing ${MODEL}" >&2; exit 1; }

BASE="http://${HOST}:${PORT}"
OWNED=0
PID=""
TMP="$(mktemp -d)"
cleanup() {
  [[ "${OWNED}" == "1" && -n "${PID}" ]] && kill "${PID}" 2>/dev/null || true
  rm -rf "${TMP}"
}
trap cleanup EXIT

if ! curl -sf "${BASE}/props" >/dev/null 2>&1; then
  echo ">>> start diffusion :${PORT} -ngl ${NGL}" >&2
  "${ZEROLLAMA_DIFFUSION_SERVER_BIN}" -m "${MODEL}" -ngl "${NGL}" -c "${CTX}" \
    --host "${HOST}" --port "${PORT}" > /tmp/dg4a-oracle.log 2>&1 &
  PID=$!
  OWNED=1
  for _ in $(seq 1 120); do
    curl -sf "${BASE}/props" >/dev/null 2>&1 && break
    kill -0 "${PID}" 2>/dev/null || { tail -40 /tmp/dg4a-oracle.log >&2; exit 1; }
    sleep 2
  done
fi

curl -sS "${BASE}/tokenize" -H 'content-type: application/json' -d '{"content":"billing"}' > "${TMP}/billing.json"
curl -sS "${BASE}/tokenize" -H 'content-type: application/json' -d '{"content":"tech"}' > "${TMP}/tech.json"
curl -sS "${BASE}/tokenize" -H 'content-type: application/json' -d '{"content":"<<dept>>"}' > "${TMP}/marker.json"

run_once() {
  local out="$1"
  python3 - "${BASE}" "${STEPS}" "${SEED}" "${TMP}" "${out}" <<'PY'
import json, sys, urllib.request
base, steps, seed, tmp, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
bill = json.load(open(f"{tmp}/billing.json"))
tech = json.load(open(f"{tmp}/tech.json"))
marker = json.load(open(f"{tmp}/marker.json"))
body = {
    "model": "openjev",
    "prompt": (
        'Reply with JSON {"dept":{"type":"choice","choice":"billing"}} '
        "then a marker line exactly:\n<<dept>> billing"
    ),
    "n_steps": steps,
    "seed": seed,
    "max_tokens": 64,
    "readout": {
        "slots": [{
            "question_id": "dept",
            "marker_tokens": marker["tokens"],
            "options": [
                {"option_id": "billing", "tokens": bill["tokens"]},
                {"option_id": "tech", "tokens": tech["tokens"]},
            ],
        }],
        "gather": [
            {"question_id": "dept", "option_id": "billing", "tokens": bill["tokens"]},
            {"question_id": "dept", "option_id": "tech", "tokens": tech["tokens"]},
        ],
    },
}
req = urllib.request.Request(base + "/v1/systemone", data=json.dumps(body).encode(),
                             headers={"content-type": "application/json"})
with urllib.request.urlopen(req, timeout=600) as r:
    open(out, "w").write(r.read().decode())
PY
}

echo ">>> oracle run A (seed=${SEED} steps=${STEPS})" >&2
run_once "${TMP}/a.json"
echo ">>> oracle run B (same seed)" >&2
run_once "${TMP}/b.json"

python3 - "${TMP}/a.json" "${TMP}/b.json" "${SEED}" "${STEPS}" <<'PY'
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
seed, steps = int(sys.argv[3]), int(sys.argv[4])
da, db = a.get("diffusion") or {}, b.get("diffusion") or {}
assert int(da.get("seed", -1)) == seed, da
assert int(db.get("seed", -1)) == seed, db
assert int(da.get("steps_requested", -1)) == steps, da
ga = {h["option_id"]: float(h["logit_mean"]) for h in (a.get("gather") or [])}
gb = {h["option_id"]: float(h["logit_mean"]) for h in (b.get("gather") or [])}
assert ga and gb and set(ga) == set(gb), (ga, gb)
best_a = max(ga, key=ga.get)
best_b = max(gb, key=gb.get)
assert best_a == best_b, f"ranking drift: {best_a} vs {best_b} logits {ga} vs {gb}"
# Soft float check — warn-level via exit 0 but print delta
max_abs = max(abs(ga[k] - gb[k]) for k in ga)
print("DG4a oracle OK:", {
    "seed": seed,
    "steps_requested": steps,
    "readout_mode": a.get("readout_mode"),
    "argmax": best_a,
    "logits_a": ga,
    "logits_b": gb,
    "max_abs_delta": max_abs,
})
if max_abs > 1e-3:
    print(f"note: float logits drifted by {max_abs:.6g} (CUDA non-bit-exact); ranking stable")
PY
