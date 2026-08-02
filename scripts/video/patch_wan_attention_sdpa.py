#!/usr/bin/env python3
"""Patch upstream Wan attention.py for WAN_FORCE_SDPA (PyTorch SDPA, skip flash_attn import).

Re-run after Wan2.1 git pull. Idempotent via marker check.
"""
from __future__ import annotations

import sys
from pathlib import Path

MARKER = "# zerollama: WAN_FORCE_SDPA sdpath fallback"

PATCHED_HEADER = '''# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import warnings

import torch

''' + MARKER + '''


def _wan_use_sdpa() -> bool:
    return os.environ.get("WAN_FORCE_SDPA", "").lower() in ("1", "true", "yes")


FLASH_ATTN_3_AVAILABLE = False
FLASH_ATTN_2_AVAILABLE = False
if not _wan_use_sdpa():
    try:
        import flash_attn_interface

        FLASH_ATTN_3_AVAILABLE = True
    except ModuleNotFoundError:
        pass
    try:
        import flash_attn

        FLASH_ATTN_2_AVAILABLE = True
    except ModuleNotFoundError:
        pass

__all__ = [
    'flash_attention',
    'attention',
]


def _sdpa_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None,
                    q_scale=None, causal=False, window_size=(-1, -1), deterministic=False,
                    dtype=torch.bfloat16):
    """PyTorch SDPA path for Wan (CPU/MPS/CUDA without flash_attn).

    Keep activation dtype (fp16/bf16/fp32). Forcing bf16 breaks Darwin float32 DiT
    (``mat1 BFloat16 / mat2 Float`` on the output projection).
    """
    _ = window_size, deterministic, softmax_scale, q_scale, q_lens, k_lens
    out_dtype = q.dtype
    compute = out_dtype if out_dtype in (torch.float16, torch.bfloat16, torch.float32) else dtype
    # [B, L, H, D] -> [B, H, L, D] for SDPA
    q = q.transpose(1, 2).to(compute)
    k = k.transpose(1, 2).to(compute)
    v = v.transpose(1, 2).to(compute)
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, is_causal=causal, dropout_p=dropout_p)
    return out.transpose(1, 2).contiguous().to(out_dtype)


'''


def patch_attention(path: Path) -> bool:
    text = path.read_text()
    if MARKER in text and "_wan_use_sdpa()" in text and "def _wan_use_sdpa() -> bool:" in text:
        if "return _wan_use_sdpa()" not in text.split("def _wan_use_sdpa")[1].split("\n")[1]:
            if "SDPBackend.MATH" in text or "enable_mem_efficient=False" in text:
                return False
            if "enable_mem_efficient=True" in text and MARKER in text:
                return False

    # Find start of flash_attention def (after upstream imports).
    marker_def = "def flash_attention("
    idx = text.find(marker_def)
    if idx == -1:
        raise SystemExit(f"flash_attention not found in {path}")

    rest = text[idx:]
    # Inject guard at start of flash_attention body if missing.
    if "if _wan_use_sdpa():" not in rest:
        rest = rest.replace(
            '    """\n    q:',
            '''    """
    q:''',
            1,
        )
        doc_end = rest.find('    half_dtypes')
        if doc_end == -1:
            raise SystemExit("flash_attention body not found")
        guard = """    if _wan_use_sdpa():
        return _sdpa_attention(
            q, k, v, q_lens=q_lens, k_lens=k_lens, dropout_p=dropout_p,
            softmax_scale=softscale, q_scale=q_scale, causal=causal,
            window_size=window_size, deterministic=deterministic, dtype=dtype)

"""
        # fix typo softmax_scale
        guard = guard.replace("softscale", "softmax_scale")
        rest = rest[:doc_end] + guard + rest[doc_end:]

    # Patch attention() tail to use _sdpa_attention
    old_else = """    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out"""
    new_else = """    else:
        return _sdpa_attention(
            q, k, v, q_lens=q_lens, k_lens=k_lens, dropout_p=dropout_p,
            softmax_scale=softmax_scale, q_scale=q_scale, causal=causal,
            window_size=window_size, deterministic=deterministic, dtype=dtype,
        )"""
    if old_else in rest:
        rest = rest.replace(old_else, new_else, 1)

    old_if = """    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention("""
    new_if = """    if not _wan_use_sdpa() and (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        return flash_attention("""
    if old_if in rest:
        rest = rest.replace(old_if, new_if, 1)

    path.write_text(PATCHED_HEADER + rest)
    return True


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path/to/Wan2.1/wan/modules/attention.py>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"not found: {path}", file=sys.stderr)
        return 1
    if patch_attention(path):
        print(f"patched {path}")
    else:
        print(f"already up to date {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
