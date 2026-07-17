#!/usr/bin/env python3
"""Download a small public instruction dataset and pre-tokenize it for ternary QAT.

Formats examples with the model's chat template (Qwen Instruct, etc.), tokenizes,
and builds labels that:
  - use -100 on padding tokens (CRITICAL: with pad_token==eos_token, copying
    input_ids into labels teaches the model to emit EOS for most positions —
    that alone produces collapsed / "character soup" generations),
  - use -100 on the prompt/user prefix so loss is response-only (standard
    instruct SFT).

Saves a `datasets.Dataset` arrow file to disk.

Usage:
    python3 scripts/qat/prepare_dataset.py \
        --model_name Qwen/Qwen2.5-0.5B-Instruct \
        --output_dir data/qat/alpaca_chat_0.5b \
        --num_examples 4000
"""

from __future__ import annotations

import argparse
import os

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


def user_content(example: dict) -> str:
    instruction = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()
    if inp:
        return f"{instruction}\n\n{inp}"
    return instruction


def build_chat_text(tokenizer, example: dict) -> tuple[str, str]:
    """Return (full_chat_text, prompt_only_text_with_generation_prompt)."""
    user = user_content(example)
    output = (example.get("output") or "").strip()
    messages_full = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": output},
    ]
    messages_prompt = [{"role": "user", "content": user}]

    # Prefer the model's native chat template when available.
    if getattr(tokenizer, "chat_template", None):
        full = tokenizer.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False
        )
        prompt = tokenizer.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )
        return full, prompt

    # Fallback Alpaca-style (should rarely hit for Instruct models).
    full = f"### Instruction:\n{user}\n\n### Response:\n{output}{tokenizer.eos_token}"
    prompt = f"### Instruction:\n{user}\n\n### Response:\n"
    return full, prompt


def main():
    parser = argparse.ArgumentParser(
        description="Prepare an instruction dataset for ternary QAT (response-only labels)."
    )
    parser.add_argument("--model_name", required=True, help="HF model id (tokenizer + chat template).")
    parser.add_argument(
        "--dataset_name",
        default="tatsu-lab/alpaca",
        help="HF dataset id (default: tatsu-lab/alpaca).",
    )
    parser.add_argument("--num_examples", type=int, default=4000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    print(f"PROGRESS:5:loading tokenizer for {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    # Prefer an explicit pad that is NOT eos when possible, so attention/labels
    # stay unambiguous. Qwen usually already defines a pad token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    print(
        f"PROGRESS:8:pad_token_id={pad_id} eos_token_id={tokenizer.eos_token_id} "
        f"pad_is_eos={pad_id == tokenizer.eos_token_id} has_chat_template={bool(tokenizer.chat_template)}"
    )

    print(f"PROGRESS:15:downloading dataset {args.dataset_name}")
    raw = load_dataset(args.dataset_name, split="train")
    if args.num_examples and args.num_examples < len(raw):
        raw = raw.shuffle(seed=args.seed).select(range(args.num_examples))

    print(f"PROGRESS:40:formatting {len(raw)} examples with chat template")
    full_texts: list[str] = []
    prompt_texts: list[str] = []
    for ex in raw:
        full, prompt = build_chat_text(tokenizer, ex)
        full_texts.append(full)
        prompt_texts.append(prompt)

    print("PROGRESS:55:tokenizing + building response-only labels")

    def tokenize_one(full: str, prompt: str) -> dict:
        full_ids = tokenizer(
            full,
            truncation=True,
            max_length=args.max_length,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]
        prompt_ids = tokenizer(
            prompt,
            truncation=True,
            max_length=args.max_length,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]

        # Response starts after the prompt prefix. If truncation ate the response,
        # keep a minimal supervised suffix so the example is not all -100.
        prompt_len = min(len(prompt_ids), max(len(full_ids) - 1, 0))

        labels = list(full_ids)
        for i in range(prompt_len):
            labels[i] = -100

        # Pad to max_length.
        pad_len = args.max_length - len(full_ids)
        if pad_len < 0:
            # Shouldn't happen given truncation, but be safe.
            full_ids = full_ids[: args.max_length]
            labels = labels[: args.max_length]
            pad_len = 0

        attention_mask = [1] * len(full_ids) + [0] * pad_len
        input_ids = full_ids + [pad_id] * pad_len
        # NEVER copy pad token ids into labels — that is what caused collapse.
        labels = labels + ([-100] * pad_len)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    rows = [tokenize_one(f, p) for f, p in zip(full_texts, prompt_texts)]
    tokenized = Dataset.from_list(rows)

    # Sanity: fraction of supervised (non -100) tokens.
    total = 0
    supervised = 0
    for row in rows[: min(200, len(rows))]:
        for t in row["labels"]:
            total += 1
            if t != -100:
                supervised += 1
    print(
        f"PROGRESS:80:label sanity on first {min(200, len(rows))} examples: "
        f"{100.0 * supervised / max(total, 1):.1f}% tokens supervised (rest -100)"
    )

    print(f"PROGRESS:85:saving to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    tokenized.save_to_disk(args.output_dir)
    print(f"PROGRESS:100:done ({len(tokenized)} examples, max_length={args.max_length})")


if __name__ == "__main__":
    main()
