"""Completion-only label masking for SFT (T9 / Unsloth-style train-on-responses).

Prompt tokens get label ``-100`` so the causal LM loss only trains on the
assistant completion. Works with chat-template, ChatML, Llama3, and Alpaca
strings produced by ``training_format``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from training_format import (
    _as_messages,
    _apply_hf_chat_template,
    format_alpaca,
    format_chatml,
    format_llama3,
    format_sft_sample,
    resolve_format_mode,
)


def format_sft_prompt_and_full(
    sample: Mapping[str, Any],
    *,
    mode: str = "auto",
    tokenizer: Any = None,
    request: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, str]:
    """Return ``(prompt_prefix, full_text)`` for completion-only masking.

    ``prompt_prefix`` is a prefix of ``full_text`` (or equal if we cannot split).
    """
    req = request or {}
    if mode == "auto":
        mode = resolve_format_mode(req, tokenizer)

    messages = _as_messages(sample)
    full = format_sft_sample(sample, mode=mode, tokenizer=tokenizer, request=req)

    if mode == "modelfile":
        # Go TEMPLATE render is full-turn only; fall back to no mask split.
        return full, full

    if mode == "hf" and tokenizer is not None:
        if messages and messages[-1].get("role") == "assistant":
            prompt_msgs = messages[:-1]
            apply = getattr(tokenizer, "apply_chat_template", None)
            if apply and getattr(tokenizer, "chat_template", None):
                try:
                    prompt = apply(
                        prompt_msgs,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    if isinstance(prompt, str) and full.startswith(prompt):
                        return prompt, full
                except Exception:
                    pass
        return full, full

    if mode == "chatml":
        if messages and messages[-1].get("role") == "assistant":
            prompt_msgs = messages[:-1]
            parts = []
            for m in prompt_msgs:
                role = m["role"] if m["role"] in ("system", "user", "assistant", "tool") else "user"
                parts.append(f"<|im_start|>{role}\n{m['content']}<|im_end|>")
            prompt = "\n".join(parts)
            if prompt:
                prompt = prompt + "\n<|im_start|>assistant\n"
            else:
                prompt = "<|im_start|>assistant\n"
            # Rebuild full to guarantee prefix relationship.
            resp = messages[-1]["content"]
            full2 = prompt + resp + "<|im_end|>"
            return prompt, full2
        return full, full

    if mode == "llama3":
        if messages and messages[-1].get("role") == "assistant":
            prompt_msgs = messages[:-1]
            parts = ["<|begin_of_text|>"]
            for m in prompt_msgs:
                role = "system" if m["role"] == "system" else (
                    "assistant" if m["role"] == "assistant" else "user"
                )
                parts.append(
                    f"<|start_header_id|>{role}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
                )
            parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            prompt = "".join(parts)
            full2 = prompt + messages[-1]["content"] + "<|eot_id|>"
            return prompt, full2
        return full, full

    if mode == "alpaca":
        prompt_body = sample.get("prompt", sample.get("instruction", sample.get("input", "")))
        response = sample.get("response", sample.get("output", sample.get("completion", "")))
        system = sample.get("system")
        if system:
            prompt = (
                f"### System:\n{system}\n\n"
                f"### Instruction:\n{prompt_body}\n\n"
                f"### Response:\n"
            )
        else:
            prompt = f"### Instruction:\n{prompt_body}\n\n### Response:\n"
        return prompt, prompt + str(response)

    return full, full


def mask_labels_completion_only(
    input_ids: Sequence[int],
    prompt_ids: Sequence[int],
) -> List[int]:
    """Set labels to -100 for the prompt prefix (token-id exact prefix match)."""
    labels = list(input_ids)
    n = 0
    plen = len(prompt_ids)
    if plen > 0 and plen <= len(input_ids) and list(input_ids[:plen]) == list(prompt_ids):
        n = plen
    for i in range(n):
        labels[i] = -100
    # At least one trainable token if possible.
    if n >= len(labels) and labels:
        labels[-1] = int(input_ids[-1])
    return labels


def tokenize_completion_only_corpus(
    training_data: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_length: int,
    mode: str,
    request: Optional[Mapping[str, Any]] = None,
) -> Dict[str, List[List[int]]]:
    """Tokenize each row with completion-only labels (no packing)."""
    req = request or {}
    input_ids_out: List[List[int]] = []
    attn_out: List[List[int]] = []
    labels_out: List[List[int]] = []
    for sample in training_data:
        prompt, full = format_sft_prompt_and_full(
            sample, mode=mode, tokenizer=tokenizer, request=req
        )
        full_enc = tokenizer(full, truncation=True, max_length=max_length, padding=False)
        ids = list(full_enc["input_ids"])
        prompt_enc = tokenizer(prompt, truncation=True, max_length=max_length, padding=False)
        pids = list(prompt_enc["input_ids"])
        # If prompt was truncated harder than full, don't mask everything.
        if len(pids) >= len(ids):
            pids = pids[: max(0, len(ids) - 1)]
        labels = mask_labels_completion_only(ids, pids)
        input_ids_out.append(ids)
        attn_out.append([1] * len(ids))
        labels_out.append(labels)
    return {
        "input_ids": input_ids_out,
        "attention_mask": attn_out,
        "labels": labels_out,
    }
