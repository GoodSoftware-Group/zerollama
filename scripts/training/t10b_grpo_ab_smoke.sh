#!/usr/bin/env bash
# T10b GRPO reward A/B smoke.
#
# Always (host / CI — no GPU):
#   ./scripts/training/t10b_grpo_ab_smoke.sh
#     → dry-run length vs format_json vs exact_match; confirm reward labels
#
# Optional GPU lab (never production ports; pick a free device):
#   RUN_E2E_T10B=1 CUDA_VISIBLE_DEVICES=1 ./scripts/training/t10b_grpo_ab_smoke.sh
#     → short grpo_run=true (max_steps=2) for length vs format_json; write timings JSON
#
# Doc: docs/t10-grpo.md · docs/t10-grpo-findings.md
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -x "${ROOT}/.venv-training/bin/python" ]]; then
  PY="${ROOT}/.venv-training/bin/python"
elif [[ -n "${TRAINING_PYTHON:-}" && -x "${TRAINING_PYTHON}" ]]; then
  PY="${TRAINING_PYTHON}"
else
  PY=python3
fi

echo "== T10b unit: reward A/B dry-run =="
"$PY" - <<'PY'
from training_grpo import process_grpo_request, resolve_reward_func

base = {
    "model_name": "sshleifer/tiny-gpt2",
    "prompts": ['{"ok": true}', "plain text"],
}
variants = [
    {"reward_builtin": "length"},
    {"reward_builtin": "format_json"},
    {
        "reward_builtin": "exact_match",
        "prompts": ["q1", "q2"],
        "references": ["a1", "a2"],
    },
]
for extra in variants:
    payload = {**base, **extra}
    # exact_match overrides prompts
    if "prompts" in extra:
        payload["prompts"] = extra["prompts"]
    res = process_grpo_request(payload)
    assert res["status"] == "ok" and res["dry_run"] is True and res["trained"] is False, res
    print("dry_run", res.get("reward"), "trl=", res.get("trl"))
    func, err = resolve_reward_func(payload)
    assert err is None and callable(func), err
print("OK: dry-run A/B labels")
PY

if [[ "${RUN_E2E_T10B:-0}" != "1" ]]; then
  echo "OK: dry-run only (set RUN_E2E_T10B=1 on a free GPU for Trainer A/B)"
  exit 0
fi

echo "== T10b lab: grpo_run A/B (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}) =="
OUT_DIR="${T10B_AB_OUT:-/tmp/zerollama-t10b-ab-$$}"
mkdir -p "$OUT_DIR"
export T10B_AB_OUT="$OUT_DIR"
# WHY default tiny-gpt2: ~disk-free smoke; needs GPT-2 LoRA targets (not LLaMA q/k/v).
# Prefer T10B_MODEL=Qwen/Qwen2.5-0.5B-Instruct on 5080/4090 when HF cache is warm.
export T10B_MODEL="${T10B_MODEL:-sshleifer/tiny-gpt2}"
export T10B_MAX_STEPS="${T10B_MAX_STEPS:-2}"

"$PY" - <<'PY'
import json, os, time, traceback
from pathlib import Path
from training_grpo import process_grpo_request, trl_grpo_available

ok, hint = trl_grpo_available()
if not ok:
    raise SystemExit(f"trl unavailable: {hint}")

out = Path(os.environ["T10B_AB_OUT"])
model = os.environ["T10B_MODEL"]
steps = int(os.environ["T10B_MAX_STEPS"])
prompts = [
    'Reply with JSON only: {"answer": 1}',
    'Reply with JSON only: {"answer": 2}',
]
# Auto GPT-2 LoRA module names when using the default toy (DEFAULT_LORA_TARGETS are LLaMA-style).
lora_targets = None
ml = model.lower()
if "gpt2" in ml or "gpt-2" in ml:
    lora_targets = ["c_attn", "c_proj"]
results = []
for builtin in ("length", "format_json"):
    dest = out / builtin
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    try:
        payload = {
            "model_name": model,
            "prompts": prompts,
            "reward_builtin": builtin,
            "grpo_run": True,
            "max_steps": steps,
            "batch_size": 2,
            "num_generations": 2,
            "use_lora": True,
            "output_dir": str(dest),
        }
        if lora_targets:
            payload["lora_target_modules"] = lora_targets
        res = process_grpo_request(payload)

    except Exception as e:
        res = {
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "trained": False,
        }
    wall = time.perf_counter() - t0
    row = {
        "reward_builtin": builtin,
        "wall_s": round(wall, 3),
        "status": res.get("status"),
        "trained": res.get("trained"),
        "error": res.get("error"),
        "output_dir": res.get("output_dir"),
        "model_name": model,
        "max_steps": steps,
    }
    results.append(row)
    print(json.dumps(row, indent=2))

summary = {"results": results, "out_dir": str(out)}
(out / "ab_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
trained = [r for r in results if r.get("trained")]
if not trained:
    # Still write summary — tip TRL / tiny model may fail; operator reads error.
    print("WARN: no trained=true run; see", out / "ab_summary.json")
    raise SystemExit(2)
print("OK: wrote", out / "ab_summary.json")
PY
