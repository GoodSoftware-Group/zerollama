# T10 / T10b — GRPO / RL job kind (stock path)

**ROADMAP:** [GPU training T10](./ROADMAP.md#phases-training-track)  
**Status (Oct 2026):** **Partial** — `kind=grpo|rl` wired; HTTP-safe rewards (builtins + `reward_url` + `reward_hf`); dry-run default; `grpo_run=true` lab-only `GRPOTrainer`; `trl>=0.14,<1` pinned in [`requirements-training.txt`](../requirements-training.txt).

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
| Deps | `trl>=0.14,<1` in [`requirements-training.txt`](../requirements-training.txt) (sync via `training_uv_venv.sh`) |
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
| `reward_builtin` | HTTP-safe builtin (see below) | Wire-safe; aliases `reward` / `reward_model` when name is a builtin |
| `reward_url` | Agent hook `http(s)://…` | POST `{completions,prompts?}` → `{rewards:[…]}`; lab agent/tool scores |
| `reward_hf` | HF seq-class id `org/name` | Lazy-load on `grpo_run`; dry-run only validates id shape |
| `references` | Parallel golds for `exact_match` | Or `training_data[].answer\|reference\|label\|response` |
| `grpo_run` | Default **false** → dry-run | CI/hosts without `trl` still validate; no accidental GPU train |
| LoRA knobs | Same as SFT when `grpo_run=true` | One PEFT path |
| `max_steps` | Default `10` for lab | Short lab loops |

### Rewards over the wire (T10b)

JSON **cannot** carry Python callables. Do **not** send `reward_funcs` / `reward_fn` as strings — validation rejects them. In-process callers of `process_grpo_request` may pass a real callable.

| Signal | Example | Notes |
|--------|---------|-------|
| Builtin `length` / `len` | `"reward_builtin": "length"` | Char length — smoke default |
| Builtin `nonempty` | `"reward_builtin": "nonempty"` | 1.0 if stripped non-empty |
| Builtin `format_json` / `json` | `"reward_builtin": "format_json"` | 1.0 if `json.loads` succeeds |
| Builtin `exact_match` / `em` | + `references` | 1.0 iff completion == reference |
| Agent URL | `"reward_url": "http://127.0.0.1:9xxx/score"` | **http(s) only** (file:// rejected) |
| HF RM | `"reward_hf": "org/reward-model"` | Also `reward_model` when value looks like `org/name` |

**WHY reject `file://` on `reward_url`:** job payloads are agent-writable; local-file schemes are an accidental read/SSRF footgun. Point the hook at a lab HTTP service you control.

### Dry-run result shape

**Why `status=ok` + `trained=false`:** the job *succeeded as a dry-run* (payload legal). Automation must check `trained` / `dry_run`, not only HTTP 200 / job COMPLETED.

```json
{
  "status": "ok",
  "kind": "grpo",
  "dry_run": true,
  "trained": false,
  "trl": false,
  "reward": "length",
  "trl_hint": "trl not installed…",
  "message": "T10 dry-run: payload OK; no training performed. …"
}
```

## Lab: GPU A/B (T10b)

On a **free** GPU (never production `:11434` / `:8081`):

1. Sync venv: `TRAINING_UV_SYNC=1 ./scripts/training/training_uv_venv.sh --verify` (pulls pinned `trl` + `requests`).
2. Dry-run + optional Trainer A/B:

```bash
./scripts/training/t10_grpo_smoke.sh
./scripts/training/t10b_grpo_ab_smoke.sh
# Trainer (pick free device):
RUN_E2E_T10B=1 CUDA_VISIBLE_DEVICES=1 ./scripts/training/t10b_grpo_ab_smoke.sh
# Prefer instruct model when cached:
# T10B_MODEL=Qwen/Qwen2.5-0.5B-Instruct RUN_E2E_T10B=1 CUDA_VISIBLE_DEVICES=1 ./scripts/training/t10b_grpo_ab_smoke.sh
```

**Lab defaults on `grpo_run`:** `num_generations=2` with aligned `generation_batch_size` (tip TRL defaults to 8 — too heavy for smokes). Sibling empty `mlx/` checkout is neutralized before Trainer (see findings).

**4090 sign-off (Oct 2026):** `length` + `format_json` both `trained=true` on tiny-gpt2 — [t10-grpo-findings.md](./t10-grpo-findings.md).

## Non-goals

- Vendoring Unsloth Core / Studio
- Production default flip for GRPO
- Blocking interactive inference without T6 idle-wait / window
- Claiming TRL tip API stability (lab may see `GRPOTrainer` kwarg drift — pin `<1` reduces that)
- Generic `run: true` alias (use `grpo_run` only — WHY: `run` collides with unrelated payload fields)
- Shipping a public internet reward proxy (operators host their own `reward_url`)

## Exit

1. Kind normalize + reject unknown (`train|run_script|grpo|rl`)
2. Host unit: `./scripts/training/t10_grpo_smoke.sh` (no `trl` / no GPU)
3. Honest dry-run (`trained=false`) vs `grpo_run=true` GPU path
4. **T10b:** pinned `trl`; builtins + `reward_url` + `reward_hf`; unit coverage for URL hook + exact_match

**Next:** optional Qwen 0.5B A/B on 5080; T11c presets/upload already shipped.
