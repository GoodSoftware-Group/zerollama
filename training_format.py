"""SFT text formatting for embedded training (no torch deps).

Why separate from training.py: unit-testable without loading PyTorch; Go/CI can
import this module alone. training.py calls format_sft_sample for each row.

Aligns train-time strings with serve-time chat templates (ROADMAP T8 / Unsloth
hygiene): Alpaca ### Instruction was a mismatch for ChatML/Llama/Qwen models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence


def _as_messages(sample: Mapping[str, Any]) -> List[Dict[str, str]]:
    """Normalize prompt/response or messages[] into chat turns."""
    if isinstance(sample.get("messages"), list) and sample["messages"]:
        out: List[Dict[str, str]] = []
        for m in sample["messages"]:
            if not isinstance(m, Mapping):
                continue
            role = str(m.get("role", "user")).strip() or "user"
            content = m.get("content", m.get("text", ""))
            out.append({"role": role, "content": str(content)})
        return out

    messages: List[Dict[str, str]] = []
    system = sample.get("system")
    if system:
        messages.append({"role": "system", "content": str(system)})
    prompt = sample.get("prompt", sample.get("instruction", sample.get("input", "")))
    response = sample.get("response", sample.get("output", sample.get("completion", "")))
    messages.append({"role": "user", "content": str(prompt)})
    messages.append({"role": "assistant", "content": str(response)})
    return messages


def format_alpaca(sample: Mapping[str, Any]) -> str:
    """Legacy ### Instruction / ### Response layout (explicit format=alpaca only)."""
    prompt = sample.get("prompt", sample.get("instruction", sample.get("input", "")))
    response = sample.get("response", sample.get("output", sample.get("completion", "")))
    system = sample.get("system")
    if system:
        return (
            f"### System:\n{system}\n\n"
            f"### Instruction:\n{prompt}\n\n"
            f"### Response:\n{response}"
        )
    return f"### Instruction:\n{prompt}\n\n### Response:\n{response}"


def format_chatml(sample: Mapping[str, Any]) -> str:
    """Qwen/ChatML-style turns — matches common serve TEMPLATEs."""
    parts: List[str] = []
    for m in _as_messages(sample):
        role = m["role"]
        if role not in ("system", "user", "assistant", "tool"):
            role = "user"
        parts.append(f"<|im_start|>{role}\n{m['content']}<|im_end|>")
    return "\n".join(parts)


def format_llama3(sample: Mapping[str, Any]) -> str:
    """Llama 3 / 3.1 instruct header layout (serve-compatible approximation)."""
    parts: List[str] = ["<|begin_of_text|>"]
    for m in _as_messages(sample):
        role = m["role"]
        if role == "system":
            hdr = "system"
        elif role == "assistant":
            hdr = "assistant"
        else:
            hdr = "user"
        parts.append(
            f"<|start_header_id|>{hdr}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
        )
    return "".join(parts)


def _apply_hf_chat_template(
    tokenizer: Any,
    sample: Mapping[str, Any],
) -> Optional[str]:
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is None:
        return None
    tmpl = getattr(tokenizer, "chat_template", None)
    if not tmpl:
        return None
    try:
        return apply(
            _as_messages(sample),
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        return None


def resolve_format_mode(
    request: Mapping[str, Any],
    tokenizer: Any = None,
) -> str:
    """Return alpaca|chatml|llama3|hf|modelfile|auto.

    auto: prefer tokenizer chat_template (hf), else chatml if tokenizer looks
    ChatML-ish, else alpaca for backward compatibility.
    """
    raw = str(request.get("format", request.get("chat_format", "auto"))).strip().lower()
    if raw in ("alpaca", "instruct", "legacy"):
        return "alpaca"
    if raw in ("chatml", "qwen", "qwen2", "qwen3"):
        return "chatml"
    if raw in ("llama3", "llama-3", "llama"):
        return "llama3"
    if raw in ("hf", "tokenizer", "chat_template"):
        return "hf"
    if raw in ("modelfile", "gotmpl", "go", "ollama-template", "template"):
        return "modelfile"
    # auto
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return "hf"
    name = ""
    if tokenizer is not None:
        name = str(getattr(tokenizer, "name_or_path", "") or "").lower()
    model_name = str(request.get("model_name", "")).lower()
    blob = name + " " + model_name
    if any(x in blob for x in ("qwen", "chatml", "yi-", "internlm")):
        return "chatml"
    if "llama-3" in blob or "llama3" in blob or "llama-3." in blob:
        return "llama3"
    # Default chatml for instruct-style HF ids; alpaca only when format=alpaca.
    # Why not alpaca default: T8 — mismatched templates hurt serve transfer.
    return "chatml"


def format_sft_sample(
    sample: Mapping[str, Any],
    *,
    mode: str = "auto",
    tokenizer: Any = None,
    request: Optional[Mapping[str, Any]] = None,
) -> str:
    """Render one training row as a plain text string for LM training."""
    req = request or {}
    if mode == "auto":
        mode = resolve_format_mode(req, tokenizer)

    if mode == "alpaca":
        return format_alpaca(sample)
    if mode == "chatml":
        return format_chatml(sample)
    if mode == "llama3":
        return format_llama3(sample)
    if mode == "modelfile":
        from training_modelfile import format_modelfile

        return format_modelfile(sample, req)
    if mode == "hf":
        text = _apply_hf_chat_template(tokenizer, sample)
        if text is not None:
            return text
        # Fall through if tokenizer lacks a usable template.
        return format_chatml(sample)
    return format_chatml(sample)


def format_sft_corpus(
    training_data: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any = None,
    request: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    req = request or {}
    mode = resolve_format_mode(req, tokenizer)
    return [
        format_sft_sample(s, mode=mode, tokenizer=tokenizer, request=req)
        for s in training_data
    ]
