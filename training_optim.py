"""T9 stock Trainer / PEFT knobs (unified backend — no Unsloth Core fork).

Why separate module: unit-testable without loading torch models; ``training.py``
applies these when building ``LoraConfig`` / ``TrainingArguments``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


DEFAULT_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# GPT-2 / some HF toys use Conv1D attention; keep as opt-in via target_modules.
GPT2_LORA_TARGETS = ["c_attn", "c_proj"]


def resolve_gradient_checkpointing(request: Dict[str, Any], *, device: str) -> bool:
    """Default on for CUDA (Unsloth-style VRAM hygiene); off on MPS/CPU."""
    if "gradient_checkpointing" in request:
        return bool(request.get("gradient_checkpointing"))
    return device == "cuda"


def resolve_gradient_accumulation_steps(request: Dict[str, Any]) -> int:
    v = int(request.get("gradient_accumulation_steps", 4) or 4)
    return max(1, v)


def resolve_optim(request: Dict[str, Any], *, use_qlora: bool, device: str) -> str:
    """Pick AdamW variant: fused on CUDA, 8-bit when QLoRA, else stock."""
    if "optim" in request and str(request.get("optim") or "").strip():
        return str(request.get("optim")).strip()
    if device != "cuda":
        return "adamw_torch"
    if use_qlora:
        # bitsandbytes 8-bit Adam pairs with QLoRA (Unsloth default family).
        return "adamw_bnb_8bit"
    return "adamw_torch_fused"


def resolve_dataloader_pin_memory(request: Dict[str, Any], *, device: str) -> bool:
    if "dataloader_pin_memory" in request:
        return bool(request.get("dataloader_pin_memory"))
    return device == "cuda"


def resolve_use_rslora(request: Dict[str, Any]) -> bool:
    """Rank-stabilized LoRA — default on (PEFT supports use_rslora)."""
    if "use_rslora" in request:
        return bool(request.get("use_rslora"))
    if "rslora" in request:
        return bool(request.get("rslora"))
    return True


def resolve_lora_dropout(request: Dict[str, Any]) -> float:
    try:
        return float(request.get("lora_dropout", 0.05))
    except (TypeError, ValueError):
        return 0.05


def resolve_lora_target_modules(request: Dict[str, Any]) -> List[str]:
    raw = request.get("lora_target_modules") or request.get("target_modules")
    if isinstance(raw, str) and raw.strip():
        return [x.strip() for x in raw.split(",") if x.strip()]
    if isinstance(raw, (list, tuple)) and raw:
        return [str(x).strip() for x in raw if str(x).strip()]
    return list(DEFAULT_LORA_TARGETS)


def resolve_torch_compile(request: Dict[str, Any]) -> bool:
    """Opt-in only — flaky under embed / some arches."""
    return bool(request.get("torch_compile", False))


def resolve_completion_only_loss(request: Dict[str, Any]) -> bool:
    """Mask prompt tokens in labels (-100). Default on for SFT hygiene."""
    if "completion_only_loss" in request:
        return bool(request.get("completion_only_loss"))
    if "train_on_responses_only" in request:
        return bool(request.get("train_on_responses_only"))
    return True


def resolve_seed(request: Dict[str, Any]) -> Optional[int]:
    if "seed" not in request or request.get("seed") is None:
        return None
    try:
        return int(request.get("seed"))
    except (TypeError, ValueError):
        return None


def build_lora_kwargs(request: Dict[str, Any], *, lora_rank: int, lora_alpha: float) -> Dict[str, Any]:
    """Keyword args for ``peft.LoraConfig`` (minus task_type)."""
    kwargs: Dict[str, Any] = {
        "r": int(lora_rank),
        "lora_alpha": float(lora_alpha),
        "lora_dropout": resolve_lora_dropout(request),
        "target_modules": resolve_lora_target_modules(request),
        "bias": "none",
        "use_rslora": resolve_use_rslora(request),
    }
    # Optional LoftQ init when PEFT + bitsandbytes support it.
    if bool(request.get("use_loftq") or request.get("loftq")):
        kwargs["init_lora_weights"] = "loftq"
        bits = int(request.get("loftq_bits", 4) or 4)
        iters = int(request.get("loftq_iter", 1) or 1)
        try:
            from peft import LoftQConfig

            kwargs["loftq_config"] = LoftQConfig(loftq_bits=bits, loftq_iter=iters)
        except Exception:
            # Older PEFT: string init may still work without loftq_config.
            pass
    return kwargs
