"""Straight-Through-Estimator (STE) ternary quantization-aware training modules.

Implements the {-1, 0, +1} weight scheme used by BitNet-style / "Bonsai"-style
1.58-bit models:
  - FP32 "shadow" weights are the trainable Parameters.
  - Forward pass: weights are grouped into chunks of `group_size` elements,
    each group gets its own FP16-precision scale `s_g = mean(abs(w_g))`,
    and each element is rounded to {-1, 0, +1} via a dead-zone threshold
    `dead_zone * s_g`. The effective weight used in the matmul/embedding
    lookup is `s_g * t_i`.
  - Backward pass: gradients flow straight through the rounding op to the
    shadow weight (identity gradient), optionally clipped to avoid exploding
    updates on saturated groups.

Only Linear/Embedding weights are ternarized; biases and norm layers are left
untouched (they are cheap and quantizing them destroys stability disproportionately).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_GROUP_SIZE = 128
# Threshold as a fraction of the per-group mean-abs scale. 0.5 is deliberately
# in the middle of typical BitNet/Bonsai configs. If groups collapse to
# all-zero (dead_zone too wide) or never hit zero (too narrow), tune this.
DEFAULT_DEAD_ZONE = 0.5
# Clip the straight-through gradient to this multiple of the group scale to
# avoid a handful of saturated/dead groups blowing up the update for the
# whole shadow weight tensor.
DEFAULT_GRAD_CLIP = 1.0


def _pad_to_group(w: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    """Flatten w and zero-pad on the right so its length is a multiple of group_size."""
    flat = w.reshape(-1)
    n = flat.numel()
    remainder = n % group_size
    if remainder == 0:
        return flat, 0
    pad = group_size - remainder
    flat = F.pad(flat, (0, pad))
    return flat, pad


def ternary_quantize(
    w: torch.Tensor,
    group_size: int = DEFAULT_GROUP_SIZE,
    dead_zone: float = DEFAULT_DEAD_ZONE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight tensor to ternary codes + per-group scales.

    Returns (effective_weight, group_scales) where effective_weight has the
    same shape/dtype as w (== group_scale * ternary_code, un-padded, reshaped
    back), and group_scales is a 1D tensor of shape [num_groups].

    This is a plain (non-autograd-tracked) function; use TernaryQuantSTE for
    the STE-wrapped autograd version used during training.
    """
    orig_shape = w.shape
    flat, pad = _pad_to_group(w.detach(), group_size)
    groups = flat.view(-1, group_size)

    scales = groups.abs().mean(dim=1, keepdim=True)  # [num_groups, 1]
    scales = scales.clamp_min(1e-8)

    threshold = dead_zone * scales
    codes = torch.zeros_like(groups)
    codes = torch.where(groups > threshold, torch.ones_like(groups), codes)
    codes = torch.where(groups < -threshold, -torch.ones_like(groups), codes)

    eff = (codes * scales).reshape(-1)
    if pad:
        eff = eff[: eff.numel() - pad]
    eff = eff.reshape(orig_shape)

    return eff, scales.squeeze(1)


class TernaryQuantSTE(torch.autograd.Function):
    """Straight-through estimator for group-wise ternary quantization.

    Forward: y = ternary_quantize(w).
    Backward: dL/dw = clip(dL/dy, -grad_clip, +grad_clip) (identity STE, optionally clipped).
    The dead_zone / group_size only affect the forward rounding; the backward
    pass is a pure (clipped) identity, which is the whole point of STE: it lets
    gradients flow through a non-differentiable rounding operation.
    """

    @staticmethod
    def forward(ctx, w: torch.Tensor, group_size: int, dead_zone: float, grad_clip: float):
        eff, _scales = ternary_quantize(w, group_size=group_size, dead_zone=dead_zone)
        ctx.grad_clip = grad_clip
        return eff

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_clip = ctx.grad_clip
        grad_input = grad_output
        if grad_clip is not None and grad_clip > 0:
            grad_input = grad_input.clamp(-grad_clip, grad_clip)
        # None gradients for group_size / dead_zone / grad_clip (non-tensor args).
        return grad_input, None, None, None


def ternary_quantize_ste(
    w: torch.Tensor,
    group_size: int = DEFAULT_GROUP_SIZE,
    dead_zone: float = DEFAULT_DEAD_ZONE,
    grad_clip: float = DEFAULT_GRAD_CLIP,
) -> torch.Tensor:
    return TernaryQuantSTE.apply(w, group_size, dead_zone, grad_clip)


@dataclass
class TernaryStats:
    """Diagnostics to catch dead-zone collapse (all-zero or all-saturated groups)."""

    frac_zero: float
    frac_pos: float
    frac_neg: float

    def is_degenerate(self, zero_hi: float = 0.98, zero_lo: float = 0.02) -> bool:
        return self.frac_zero > zero_hi or self.frac_zero < zero_lo


def ternary_stats(w: torch.Tensor, group_size: int = DEFAULT_GROUP_SIZE, dead_zone: float = DEFAULT_DEAD_ZONE) -> TernaryStats:
    flat, pad = _pad_to_group(w.detach(), group_size)
    groups = flat.view(-1, group_size)
    scales = groups.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
    threshold = dead_zone * scales
    pos = (groups > threshold).float().mean().item()
    neg = (groups < -threshold).float().mean().item()
    zero = 1.0 - pos - neg
    return TernaryStats(frac_zero=zero, frac_pos=pos, frac_neg=neg)


class TernaryLinear(nn.Module):
    """Drop-in replacement for nn.Linear with ternary shadow-weight QAT.

    The FP32 shadow weight is the trainable Parameter (`self.weight`); the
    forward pass runs the STE ternary quantizer over it before the matmul.
    Bias, if present, stays full precision and unquantized.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = DEFAULT_GROUP_SIZE,
        dead_zone: float = DEFAULT_DEAD_ZONE,
        grad_clip: float = DEFAULT_GRAD_CLIP,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.dead_zone = dead_zone
        self.grad_clip = grad_clip
        self.frozen = False  # once True, self.weight already holds final ternary values

        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = DEFAULT_GROUP_SIZE,
        dead_zone: float = DEFAULT_DEAD_ZONE,
        grad_clip: float = DEFAULT_GRAD_CLIP,
    ) -> "TernaryLinear":
        module = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            group_size=group_size,
            dead_zone=dead_zone,
            grad_clip=grad_clip,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            module.weight.copy_(linear.weight)
            if linear.bias is not None:
                module.bias.copy_(linear.bias)
        return module

    def effective_weight(self) -> torch.Tensor:
        if self.frozen:
            return self.weight
        return ternary_quantize_ste(self.weight, self.group_size, self.dead_zone, self.grad_clip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.effective_weight()
        return F.linear(x, w, self.bias)

    @torch.no_grad()
    def freeze(self):
        """Materialize final ternary weight (scale * code) and stop STE passthrough."""
        eff, _ = ternary_quantize(self.weight, self.group_size, self.dead_zone)
        self.weight.copy_(eff)
        self.frozen = True

    def stats(self) -> TernaryStats:
        return ternary_stats(self.weight, self.group_size, self.dead_zone)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, group_size={self.group_size}, frozen={self.frozen}"


class TernaryEmbedding(nn.Module):
    """Drop-in replacement for nn.Embedding with ternary shadow-weight QAT.

    If `tied_linear` is provided, this embedding's ternarized weight (transposed)
    is reused as the LM head instead of creating an independently-drifting
    ternary weight for the head.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        group_size: int = DEFAULT_GROUP_SIZE,
        dead_zone: float = DEFAULT_DEAD_ZONE,
        grad_clip: float = DEFAULT_GRAD_CLIP,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.group_size = group_size
        self.dead_zone = dead_zone
        self.grad_clip = grad_clip
        self.frozen = False

        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.normal_(self.weight)

    @classmethod
    def from_embedding(
        cls,
        emb: nn.Embedding,
        group_size: int = DEFAULT_GROUP_SIZE,
        dead_zone: float = DEFAULT_DEAD_ZONE,
        grad_clip: float = DEFAULT_GRAD_CLIP,
    ) -> "TernaryEmbedding":
        module = cls(
            emb.num_embeddings,
            emb.embedding_dim,
            padding_idx=emb.padding_idx,
            group_size=group_size,
            dead_zone=dead_zone,
            grad_clip=grad_clip,
            device=emb.weight.device,
            dtype=emb.weight.dtype,
        )
        with torch.no_grad():
            module.weight.copy_(emb.weight)
        return module

    def effective_weight(self) -> torch.Tensor:
        if self.frozen:
            return self.weight
        return ternary_quantize_ste(self.weight, self.group_size, self.dead_zone, self.grad_clip)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        w = self.effective_weight()
        return F.embedding(input_ids, w, padding_idx=self.padding_idx)

    def lm_head_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Used when tie_word_embeddings=True: reuse this ternarized weight transposed."""
        w = self.effective_weight()
        return F.linear(hidden_states, w)

    @torch.no_grad()
    def freeze(self):
        eff, _ = ternary_quantize(self.weight, self.group_size, self.dead_zone)
        self.weight.copy_(eff)
        self.frozen = True

    def stats(self) -> TernaryStats:
        return ternary_stats(self.weight, self.group_size, self.dead_zone)

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, group_size={self.group_size}, frozen={self.frozen}"


class TiedTernaryLMHead(nn.Module):
    """LM head that reuses a TernaryEmbedding's ternarized weight transposed.

    Used instead of a second, independently-drifting TernaryLinear when
    config.tie_word_embeddings is True.
    """

    def __init__(self, tied_embedding: TernaryEmbedding, bias: nn.Parameter | None = None):
        super().__init__()
        self.tied_embedding = tied_embedding
        self.bias = bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.tied_embedding.lm_head_forward(hidden_states)
        if self.bias is not None:
            logits = logits + self.bias
        return logits

    def extra_repr(self) -> str:
        return "tied=True"


# Submodule name fragments that identify the projections we want to ternarize.
# Covers Llama/Qwen/Mistral-family naming; MLP "gate/up/down" is SwiGLU-style.
_TARGET_LINEAR_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def replace_with_ternary(
    model: nn.Module,
    include_lm_head: bool = True,
    include_embeddings: bool = True,
    group_size: int = DEFAULT_GROUP_SIZE,
    dead_zone: float = DEFAULT_DEAD_ZONE,
    grad_clip: float = DEFAULT_GRAD_CLIP,
) -> nn.Module:
    """Walk named modules and swap attention/MLP projections (+ optional
    embeddings / LM head) for ternary STE equivalents. RMSNorm/LayerNorm and
    biases are left untouched.

    Handles weight tying: if model.config.tie_word_embeddings is True and
    embeddings are ternarized, the LM head is replaced with a TiedTernaryLMHead
    that reuses the ternarized input-embedding weight transposed.
    """
    config = getattr(model, "config", None)
    tie_embeddings = bool(getattr(config, "tie_word_embeddings", False))

    # 1) Replace attention/MLP Linear projections in-place, by walking parents.
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and child_name in _TARGET_LINEAR_SUFFIXES:
                ternary = TernaryLinear.from_linear(
                    child, group_size=group_size, dead_zone=dead_zone, grad_clip=grad_clip
                )
                setattr(parent, child_name, ternary)

    # 2) Optionally replace the token embedding.
    ternary_embed_module = None
    if include_embeddings:
        embed_tokens = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        if isinstance(embed_tokens, nn.Embedding):
            ternary_embed_module = TernaryEmbedding.from_embedding(
                embed_tokens, group_size=group_size, dead_zone=dead_zone, grad_clip=grad_clip
            )
            model.set_input_embeddings(ternary_embed_module)

    # 3) Optionally replace / tie the LM head.
    if include_lm_head and hasattr(model, "lm_head"):
        old_head = model.lm_head
        if tie_embeddings and ternary_embed_module is not None:
            bias = getattr(old_head, "bias", None)
            model.lm_head = TiedTernaryLMHead(ternary_embed_module, bias=bias)
        elif isinstance(old_head, nn.Linear):
            model.lm_head = TernaryLinear.from_linear(
                old_head, group_size=group_size, dead_zone=dead_zone, grad_clip=grad_clip
            )

    return model


def iter_ternary_modules(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, (TernaryLinear, TernaryEmbedding)):
            yield name, module


def freeze_all_ternary(model: nn.Module) -> dict[str, TernaryStats]:
    """Materialize final ternary weights everywhere (end of training)."""
    stats: dict[str, TernaryStats] = {}
    for name, module in iter_ternary_modules(model):
        stats[name] = module.stats()
        module.freeze()
    return stats


@torch.no_grad()
def unwrap_ternary_to_plain(model: nn.Module) -> nn.Module:
    """Replace frozen Ternary modules with plain nn.Linear/nn.Embedding holding
    the same (already-materialized scale*code) weights, and restore standard
    tied-embedding wiring (config.tie_word_embeddings) instead of
    TiedTernaryLMHead.

    Call this after freeze_all_ternary(), right before save_pretrained(), so
    the resulting HF checkpoint has ordinary tensor names/shapes that
    convert_hf_to_gguf.py understands with no custom-architecture handling.
    The ternary-ness of the weights (exactly {-1,0,+1} * per-group scale) is
    preserved -- only the module *type* changes back to vanilla PyTorch.
    """
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, TernaryLinear):
                plain = nn.Linear(
                    child.in_features, child.out_features, bias=child.bias is not None,
                    device=child.weight.device, dtype=child.weight.dtype,
                )
                plain.weight.copy_(child.weight)
                if child.bias is not None:
                    plain.bias.copy_(child.bias)
                setattr(parent, child_name, plain)
            elif isinstance(child, TernaryEmbedding):
                plain = nn.Embedding(
                    child.num_embeddings, child.embedding_dim, padding_idx=child.padding_idx,
                    device=child.weight.device, dtype=child.weight.dtype,
                )
                plain.weight.copy_(child.weight)
                setattr(parent, child_name, plain)

    if hasattr(model, "lm_head") and isinstance(model.lm_head, TiedTernaryLMHead):
        embed = model.get_input_embeddings()
        head = nn.Linear(embed.embedding_dim, embed.num_embeddings, bias=model.lm_head.bias is not None)
        head.weight = embed.weight  # tie, matching config.tie_word_embeddings
        if model.lm_head.bias is not None:
            head.bias.copy_(model.lm_head.bias)
        model.lm_head = head

    return model
