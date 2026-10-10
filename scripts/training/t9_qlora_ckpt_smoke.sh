#!/usr/bin/env bash
# T9 stock Trainer smoke: completion-only packing masks + QLoRA + grad checkpoint.
#
# Always (fast, no GPU):
#   ./scripts/training/t9_qlora_ckpt_smoke.sh
#     → unittest pack / labels / optim
#
# Optional GPU (never production ports):
#   RUN_E2E_T9=1 CUDA_VISIBLE_DEVICES=1 ./scripts/training/t9_qlora_ckpt_smoke.sh
#     → tiny QLoRA + gradient_checkpointing train step (needs .venv-training)
#
# Doc: docs/gpu-training.md · ROADMAP T9
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

echo "== T9 unit tests (pack-aware completion masks) =="
"$PY" -m unittest \
  tests.test_training_pack \
  tests.test_training_optim \
  -v

if [[ "${RUN_E2E_T9:-0}" != "1" ]]; then
  echo "OK: unit tests only (set RUN_E2E_T9=1 for QLoRA+ckpt GPU smoke)"
  exit 0
fi

if [[ -x "${ROOT}/.venv-training/bin/python" ]]; then
  PY="${ROOT}/.venv-training/bin/python"
fi

echo "== T9 GPU: QLoRA + gradient_checkpointing (device=${CUDA_VISIBLE_DEVICES:-all}) =="
exec "$PY" - <<'PY'
from __future__ import annotations

import os
import tempfile

try:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    # WHY neutralize repo-root mlx/: it is mlx-cgo stubs (namespace package), not
    # Apple MLX. transformers 5.x caches is_mlx_available() True via find_spec,
    # then is_tensor → mlx.core ImportError on the train step (Linux CUDA).
    import transformers.utils.generic as _tf_gen
    import transformers.utils.import_utils as _tf_iu

    _tf_iu.is_mlx_available.cache_clear()
    _tf_iu.is_mlx_available = lambda: False  # type: ignore[assignment]
    _tf_gen._is_mlx_available = False
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )
except ImportError as e:
    print(f"SKIP GPU: missing training deps ({e})")
    print("  fix: TRAINING_UV_VENV=/mnt/nvme/zerollama/.venv-training \\")
    print("       ./scripts/training/training_uv_venv.sh --verify")
    raise SystemExit(0)

if not torch.cuda.is_available():
    print("SKIP: no CUDA")
    raise SystemExit(0)

print("device:", torch.cuda.get_device_name(0), flush=True)

from training_labels import tokenize_and_pack_completion_only
from training_optim import resolve_gradient_checkpointing, resolve_optim
from training_pack import packing_stats

# Tiny local-or-hub model; operators may override T9_MODEL.
model_id = os.environ.get("T9_MODEL", "sshleifer/tiny-gpt2")
print("model:", model_id, flush=True)

req = {
    "use_qlora": True,
    "use_lora": True,
    "gradient_checkpointing": True,
    "completion_only_loss": True,
    "packing": True,
    "max_length": 128,
    "max_steps": 2,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-4,
    "format": "alpaca",
    "lora_rank": 4,
    "lora_alpha": 8,
}
assert resolve_gradient_checkpointing(req, device="cuda") is True
assert resolve_optim(req, use_qlora=True, device="cuda") == "adamw_bnb_8bit"

data = [
    {"prompt": "Name a color", "response": "Blue"},
    {"prompt": "Name a fruit", "response": "Apple"},
    {"prompt": "Name a planet", "response": "Mars"},
]

tok = AutoTokenizer.from_pretrained(model_id)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

packed = tokenize_and_pack_completion_only(
    data, tok, max_length=128, mode="alpaca", request=req
)
stats = packing_stats(len(data), packed)
assert any(-100 in row for row in packed["labels"]), "expected prompt masks in packed labels"
print("pack:", stats, flush=True)

try:
    import bitsandbytes  # noqa: F401
except ImportError as e:
    print(f"SKIP GPU: bitsandbytes required for QLoRA ({e})")
    raise SystemExit(0)

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb,
    device_map={"": 0},
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(
    model,
    LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn"] if "gpt2" in model_id.lower() else ["q_proj", "v_proj"],
    ),
)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

ds = Dataset.from_dict(packed)
with tempfile.TemporaryDirectory(prefix="t9-smoke-") as out:
    args = TrainingArguments(
        output_dir=out,
        max_steps=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        logging_steps=1,
        report_to=[],
        remove_unused_columns=False,
        fp16=True,
        optim="adamw_bnb_8bit",
    )
    # transformers≥4.46: processing_class; older: tokenizer=
    try:
        trainer = Trainer(model=model, args=args, train_dataset=ds, processing_class=tok)
    except TypeError:
        trainer = Trainer(model=model, args=args, train_dataset=ds, tokenizer=tok)
    result = trainer.train()
    print("train_loss:", getattr(result, "training_loss", None), flush=True)
    print("OK: T9 QLoRA+ckpt smoke", flush=True)
PY
