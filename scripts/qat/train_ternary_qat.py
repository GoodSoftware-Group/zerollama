#!/usr/bin/env python3
"""Ternary (1.58-bit) quantization-aware training entrypoint.

Standalone script — run directly, e.g.:

    PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/qat/train_ternary_qat.py \
        --model_name Qwen/Qwen2.5-0.5B-Instruct \
        --dataset_path data/qat/alpaca_qwen25_0.5b \
        --output_dir out/qat/qwen25_0.5b_ternary \
        --num_epochs 1 --batch_size 4 --learning_rate 2e-5

Loads a causal LM, replaces attention/MLP projections + embeddings (+ LM
head, tied or not) with ternary STE modules from ternary_ste.py, trains with
HF Trainer on MPS (Apple Silicon GPU) if available, then freezes the ternary
weights (materializing scale * ternary_code, no more STE passthrough) before
saving a normal HF checkpoint that convert_hf_to_gguf.py can consume as-is.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ternary_ste import (  # noqa: E402
    freeze_all_ternary,
    iter_ternary_modules,
    replace_with_ternary,
    unwrap_ternary_to_plain,
)


def detect_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ProgressCallback(TrainerCallback):
    """Emits PROGRESS:NN:message lines so callers can track training without
    parsing HF Trainer's default logging."""

    def __init__(self, total_steps: int):
        self.total_steps = max(total_steps, 1)

    def on_train_begin(self, args, state, control, **kwargs):
        print("PROGRESS:0:training started", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        pct = min(int(100 * state.global_step / self.total_steps), 99)
        print(f"PROGRESS:{pct}:step {state.global_step}/{self.total_steps} loss={logs['loss']:.4f}", flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        print("PROGRESS:99:training loop finished, freezing ternary weights", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Ternary QAT training for small causal LMs.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dataset_path", required=True, help="Path to a datasets.Dataset saved via save_to_disk.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--dead_zone", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--include_lm_head", type=lambda s: s.lower() != "false", default=True)
    parser.add_argument(
        "--include_embeddings",
        type=lambda s: s.lower() != "false",
        default=True,
        help="Ternarize input embeddings (and tied LM head when applicable).",
    )
    parser.add_argument("--device", default="auto", help="auto|mps|cpu")
    parser.add_argument("--max_steps", type=int, default=-1, help="Override epoch-based step count for quick smoke runs.")
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = detect_device(args.device)
    print(f"PROGRESS:1:using device={device}", flush=True)

    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    # MPS does not reliably support bf16 training the way CUDA does; fp32
    # shadow weights are the safest default (this also matches the STE
    # design, where full-precision shadow weights are the point).
    torch_dtype = torch.float32

    print(f"PROGRESS:5:loading tokenizer/model {args.model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True, torch_dtype=torch_dtype
    )

    print("PROGRESS:15:replacing Linear/Embedding layers with ternary STE modules", flush=True)
    replace_with_ternary(
        model,
        include_lm_head=args.include_lm_head,
        include_embeddings=args.include_embeddings,
        group_size=args.group_size,
        dead_zone=args.dead_zone,
        grad_clip=args.grad_clip,
    )
    num_ternary = sum(1 for _ in iter_ternary_modules(model))
    print(f"PROGRESS:20:{num_ternary} ternary modules installed", flush=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model.to(device)

    print(f"PROGRESS:25:loading dataset from {args.dataset_path}", flush=True)
    from datasets import load_from_disk

    dataset = load_from_disk(args.dataset_path)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    steps_per_epoch = max(1, len(dataset) // args.batch_size)
    total_steps = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.num_epochs)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "_trainer_tmp"),
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_strategy="no",
        save_total_limit=1,
        report_to=[],
        use_cpu=(device == "cpu"),
        fp16=False,
        bf16=False,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[ProgressCallback(total_steps)],
    )

    print("PROGRESS:30:starting training loop", flush=True)
    trainer.train()

    print("PROGRESS:95:materializing final ternary weights (freezing STE)", flush=True)
    stats = freeze_all_ternary(model)
    degenerate = {name: s for name, s in stats.items() if s.is_degenerate()}
    if degenerate:
        print(f"PROGRESS:97:warning: {len(degenerate)} modules have degenerate ternary distributions "
              f"(mostly-zero or mostly-saturated) -- consider tuning --dead_zone", flush=True)
        for name, s in list(degenerate.items())[:10]:
            print(f"  degenerate: {name} frac_zero={s.frac_zero:.3f} frac_pos={s.frac_pos:.3f} frac_neg={s.frac_neg:.3f}", flush=True)

    print(f"PROGRESS:98:saving HF checkpoint to {args.output_dir}", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)
    model.to("cpu")
    # Swap Ternary modules back to plain nn.Linear/nn.Embedding (weights are
    # already frozen scale*code values) so convert_hf_to_gguf.py sees a
    # normal architecture with no custom tensor names to map.
    unwrap_ternary_to_plain(model)

    # Quick coherence smoke before save (catches pad-label collapse early).
    try:
        model.eval()
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Say hello."}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=48, do_sample=False)
        gen = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        print(f"PROGRESS:98:smoke_generate: {gen!r}", flush=True)
    except Exception as exc:  # noqa: BLE001 — smoke only
        print(f"PROGRESS:98:smoke_generate_failed: {exc}", flush=True)

    model.save_pretrained(args.output_dir, safe_serialization=False)
    tokenizer.save_pretrained(args.output_dir)

    # Clean up the transient Trainer output dir (checkpoints we didn't want to keep).
    tmp_dir = os.path.join(args.output_dir, "_trainer_tmp")
    if os.path.isdir(tmp_dir):
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("PROGRESS:100:done", flush=True)


if __name__ == "__main__":
    main()
