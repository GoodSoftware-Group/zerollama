"""Padding-free + packing collators for causal LM SFT (TRL / Unsloth-style).

- **padding_free** (default on): ``DataCollatorWithFlattening`` concatenates the
  mini-batch into one long sequence ``[1, total_tokens]`` with per-sample
  ``position_ids`` and ``labels`` separators (``-100``). No pad tokens → no
  wasted matmul on pads. Requires the model to honor ``position_ids`` (stock
  HF causal LMs do).
- **packing** (opt-in): dataset-level concat of short rows into ``max_length``
  blocks (see ``training_pack.py``) before the collator runs.

Why not Unsloth Triton isolation: we stay on Transformers/Trainer; flattening
matches TRL ``padding_free=True``. Document-isolated flash-attn kwargs are
optional when the installed transformers supports ``return_flash_attn_kwargs``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def resolve_padding_free(request: Dict[str, Any]) -> bool:
    """Default True (Unsloth/TRL hygiene). Set padding_free=false to pad batches."""
    if "padding_free" in request:
        return bool(request.get("padding_free"))
    # Aliases
    if "padding" in request:
        p = str(request.get("padding")).strip().lower()
        if p in ("free", "none", "false", "0", "off"):
            return p in ("free", "none")
        if p in ("longest", "max_length", "true", "1", "on"):
            return False
    return True


def resolve_packing(request: Dict[str, Any]) -> bool:
    return bool(request.get("packing", False))


def build_sft_collator(
    tokenizer: Any,
    *,
    padding_free: bool = True,
    flash_attn_kwargs: bool = False,
) -> Tuple[Any, str]:
    """Return (collator, mode_name).

    mode_name is ``flattening`` | ``longest`` for logs/job results.
    """
    if padding_free:
        try:
            from transformers import DataCollatorWithFlattening

            kwargs: Dict[str, Any] = {
                "return_position_ids": True,
                "separator_id": -100,
            }
            # Newer transformers: pass flash-attn cu_seqlens when requested.
            try:
                collator = DataCollatorWithFlattening(
                    return_flash_attn_kwargs=bool(flash_attn_kwargs),
                    **kwargs,
                )
            except TypeError:
                collator = DataCollatorWithFlattening(**kwargs)
                if flash_attn_kwargs:
                    raise RuntimeError(
                        "padding_free_flash_attn requires transformers with "
                        "DataCollatorWithFlattening(return_flash_attn_kwargs=...)"
                    )
            mode = "flattening_flash" if flash_attn_kwargs else "flattening"
            return collator, mode
        except RuntimeError:
            raise
        except Exception:
            pass

    from transformers import DataCollatorForLanguageModeling

    return (
        DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        "longest",
    )
