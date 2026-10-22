# T10 — GRPO / RL job kind (stock path)

**ROADMAP:** [GPU training T10](./ROADMAP.md#phases-training-track)  
**Status (Oct 2026):** **Partial (scaffold)** — `kind=grpo|rl` wired; HTTP-safe reward builtins; dry-run default (no `trl` required); `grpo_run=true` lab-only `GRPOTrainer` (requires `trl`).

## Why this exists

| Temptation | Why we do not |
|------------|----------------|
| Add `ZEROLLAMA_TRAIN_BACKEND=unsloth` | Second stack to pin, debug, and VRAM-broker; Unsloth Core is not our product |
| Vendor Unsloth Studio | **AGPL-3.0** UI; operators who need Studio stay external → GGUF → `pull` |
| Default-on GRPO in production | Needs reward design + `trl` tip stability; wrong surprise for `/api/train` clients |
| Accept `reward_funcs` over JSON | Wire format cannot carry Python callables — pretending otherwise causes silent no-ops |

**Product shape:** same embed queue + T6 idle-wait / defer + VRAM broker as SFT. Agents get a **job kind**, not a second daemon.

## Shape

| Piece | Detail |
|-------|--------|
| HTTP | `POST /api/train/jobs` with `"kind": "grpo"` or `"rl"` |
| Embed cmd | `grpo` (worker owns `training_active` — WHY: OccupiesGPU / block-inference must stay armed for the whole job) |
| Module | [`training_grpo.py`](../training_grpo.py) |
| Kind normalize | [`server/training_kind.go`](../server/training_kind.go) — unknown kinds **400** (WHY: silent coerce-to-`train` hid typos) |
| Smoke | [`scripts/training/t10_grpo_smoke.sh`](../scripts/training/t10_grpo_smoke.sh) |

## Payload (minimum)

```json
{
  "kind": "grpo",
  "payload": {
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
    "prompts": ["Write a haiku about GPUs"],
    "reward_builtin": "length",
    "output_dir": "/tmp/grpo_out"
  }
}
```

| Field | Role | WHY |
|-------|------|-----|
| `prompts` or `training_data[].prompt` | Prompt list | Same row shape operators already use for SFT |
| `reward_builtin` | HTTP-safe builtin — today only `length` | Wire-safe; aliases `reward`/`reward_model`=`"length"` |
| `grpo_run` | Default **false** → dry-run | CI/hosts without `trl` still validate; no accidental GPU train |
| LoRA knobs | Same as SFT when `grpo_run=true` | One PEFT path |
| `max_steps` | Default `10` for lab | Short lab loops |

### Rewards over the wire

JSON **cannot** carry Python callables. Do **not** send `reward_funcs` / `reward_fn` as strings — validation rejects them. In-process callers of `process_grpo_request` may pass a real callable. HF reward models = **T10b**.

### Dry-run result shape

**Why `status=ok` + `trained=false`:** the job *succeeded as a dry-run* (payload legal). Automation must check `trained` / `dry_run`, not only HTTP 200 / job COMPLETED.

```json
{
  "status": "ok",
  "kind": "grpo",
  "dry_run": true,
  "trained": false,
  "trl": false,
  "trl_hint": "trl not installed…",
  "message": "T10 dry-run: payload OK; no training performed. …"
}
```

## Non-goals

- Vendoring Unsloth Core / Studio
- Production default flip for GRPO
- Blocking interactive inference without T6 idle-wait / window
- Claiming TRL tip API stability (lab may see `GRPOTrainer` kwarg drift)
- Generic `run: true` alias (use `grpo_run` only — WHY: `run` collides with unrelated payload fields)

## Exit (scaffold — this doc)

1. Kind normalize + reject unknown (`train|run_script|grpo|rl`)
2. Host unit: `./scripts/training/t10_grpo_smoke.sh` (no `trl` / no GPU)
3. Honest dry-run (`trained=false`) vs `grpo_run=true` GPU path

**Next (T10b):** agent reward hooks; HF reward models; 5080 A/B; pin `trl` in training extras.
