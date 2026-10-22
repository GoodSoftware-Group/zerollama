# T10 / T10b findings (GRPO stock path)

Short operator notes. Spec: [t10-grpo.md](./t10-grpo.md).

## Decisions

| Choice | WHY |
|--------|-----|
| Dry-run default | CI / hosts without `trl` still validate; `grpo_run` is the only Trainer gate |
| No bare `run` flag | Collides with unrelated payload fields |
| Pin `trl>=0.14,<1` + `requests` | Floor for `GRPOTrainer`; tip 0.29 imports `requests` via vllm_client |
| Lab `num_generations=2` / aligned `generation_batch_size` | Tip default 8/8 fails tiny smokes (`batch_size` must divide) |
| Builtins over stringified callables | JSON cannot carry Python; reject `reward_funcs` strings loudly |
| `reward_url` http(s) only | Agent-writable jobs + `file://` = local-read footgun |
| HF via `reward_hf` / `org/name` | Lazy-load on Trainer path; dry-run only checks id shape |
| Worker owns `training_active` | Clearing inside `training_grpo` raced OccupiesGPU / block-inference |
| Sibling `mlx/` shield | Empty repo-root `mlx/` namespace makes transformers `is_mlx_available()` True without `mlx.core` → mid-step crash |

## Lab A/B (4090 CT, Oct 2026)

Host: dual 4090, `CUDA_VISIBLE_DEVICES=1`, `trl==0.29.1`, toy `sshleifer/tiny-gpt2` + GPT-2 LoRA targets `c_attn`/`c_proj`, `max_steps=2`, `num_generations=2`.

| reward_builtin | wall_s | trained | notes |
|----------------|--------|---------|-------|
| `length` | ~2.4 | yes | reward mean ~1.6k (char length on clipped completions) |
| `format_json` | ~1.1 | yes | reward mean **0** on tiny-gpt2 (does not emit JSON — expected) |

Smoke: `./scripts/training/t10b_grpo_ab_smoke.sh` (dry-run always); `RUN_E2E_T10B=1 CUDA_VISIBLE_DEVICES=1 …` for Trainer.

### T10c — Qwen2.5-0.5B-Instruct (4090 #1, Oct 2026)

`T10B_MODEL=Qwen/Qwen2.5-0.5B-Instruct` (default LLaMA-style LoRA targets), same `max_steps=2` / `num_generations=2`.

| reward_builtin | wall_s | trained | notes |
|----------------|--------|---------|-------|
| `length` | ~23.1 | yes | includes cold load; reward mean ~850–1k on clipped 256-tok completions |
| `format_json` | ~8.1 | yes | reward mean **0** (short JSON prompts; model does not emit JSON — expected wire check) |

Artifact: `/tmp/zerollama-t10c-qwen-ab/ab_summary.json`.

### T10c — Qwen2.5-0.5B-Instruct (5080 CT 1564, Oct 2026)

Prod tip taken down briefly; `CUDA_VISIBLE_DEVICES=0`, production `.venv-training` (`trl==0.29.1`), `max_steps=2` / `num_generations=2`. Binary linked **libpython3.11** (must match venv — see below).

| reward_builtin | wall_s | trained | notes |
|----------------|--------|---------|-------|
| `length` | ~14.1 | yes | warmer than 4090 cold (~23s); reward mean ~1.1–1.2k |
| `format_json` | ~10.6 | yes | reward mean **0** (same expected wire check) |

Artifact: `/tmp/zerollama-t10b-ab-*/ab_summary.json`. **T9** on same card: `RUN_E2E_T9=1` QLoRA+ckpt tiny-gpt2 PASS.

**Ops traps closed on this run:**

- `TRAINING_UV_SYNC=1` + cu128 index: uv first-index kept `requests==2.28.1` — `training_uv_venv.sh` now passes `--index-strategy unsafe-best-match`.
- Embed vs venv ABI: `go build` without `source ./scripts/training/training_embed_build_env.sh 3.11` links 3.10 → numpy/trl import fails in-process.

## Lab checklist

- [x] `trl` + `requests` in `.venv-training`
- [x] Dry-run `length` / `format_json` / `exact_match`
- [x] `grpo_run=true` max_steps≤2 on free GPU → `trained=true`
- [x] A/B wall time recorded (above)
- [x] Qwen 0.5B instruct A/B on 4090 (T10c)
- [x] 5080 re-run with production `.venv-training` (T10c + T9)

## Non-claims

- Not Unsloth Core / Studio parity
- Not production-default GRPO
- Not a hosted reward service
- Toy GPT-2 is for Trainer wire only — not quality signal
