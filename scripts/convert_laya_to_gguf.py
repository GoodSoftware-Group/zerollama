#!/usr/bin/env python3
"""Convert a Laya HF checkpoint (ModernBERT + decision head) to GGUF arch=laya.

Requires sibling llama.cpp gguf-py on PYTHONPATH, e.g.:

  PYTHONPATH=../llama.cpp \\
    python3 scripts/convert_laya_to_gguf.py \\
      --model ~/.cache/huggingface/hub/.../laya \\
      --outfile laya-f16.gguf

English ModernBERT-large layout expected. Head tensor names follow llama.cpp
LLM_ARCH_LAYA (laya.type_embd, laya.head.%d.*, laya.scorer.*, laya.act.*).

WHY arch=laya (not modern-bert): RANK/embed pooling cannot expose per-MASK logits
+ act head. WHY key names: SWA RoPE must be laya.rope.freq_base_swa (not
laya.attention.rope.…); silent GGUF key misses corrupt SWA RoPE. Prefer cfg n_act
when set. Docs: docs/laya-llama-cpp.md · findings.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    load_safetensors = None

try:
    import gguf
except ImportError as e:
    raise SystemExit(
        "gguf package required — set PYTHONPATH to sibling llama.cpp "
        f"(e.g. PYTHONPATH=../llama.cpp). Import error: {e}"
    ) from e


def _load_weights(model_dir: Path) -> dict[str, np.ndarray]:
    st = model_dir / "model.safetensors"
    if not st.exists():
        raise FileNotFoundError(f"missing {st}")
    if load_safetensors is None:
        raise SystemExit("pip install safetensors torch")
    tensors = load_safetensors(str(st), device="cpu")
    out = {}
    for k, v in tensors.items():
        out[k] = v.detach().cpu().numpy()
    return out


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _add_encoder(writer: gguf.GGUFWriter, weights: dict, enc_cfg: dict) -> None:
    n_layer = int(enc_cfg["num_hidden_layers"])
    n_embd = int(enc_cfg["hidden_size"])
    n_ff = int(enc_cfg.get("intermediate_size", 4 * n_embd))
    n_head = int(enc_cfg["num_attention_heads"])
    n_ctx = int(enc_cfg.get("max_position_embeddings", 512))
    vocab = int(enc_cfg.get("vocab_size", 50368))
    eps = float(enc_cfg.get("layer_norm_eps", 1e-5))
    rope_theta = float(enc_cfg.get("global_rope_theta", enc_cfg.get("rope_theta", 160000)))

    writer.add_name("laya")
    writer.add_block_count(n_layer)
    writer.add_context_length(n_ctx)
    writer.add_embedding_length(n_embd)
    writer.add_feed_forward_length(n_ff)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head)
    writer.add_layer_norm_eps(eps)
    writer.add_rope_freq_base(rope_theta)
    try:
        writer.add_file_type(gguf.LlamaFileType.ALL_F32)
    except Exception:
        pass

    # Map HF ModernBERT / Laya encoder.* → GGUF modern-bert-style names under arch laya.
    def w(name: str) -> np.ndarray:
        if name not in weights:
            raise KeyError(name)
        return weights[name]

    # embeddings
    te = w("encoder.embeddings.tok_embeddings.weight")
    writer.add_tensor("token_embd.weight", te.astype(np.float32))
    tn = w("encoder.embeddings.norm.weight")
    writer.add_tensor("token_embd_norm.weight", tn.astype(np.float32))

    for i in range(n_layer):
        p = f"encoder.layers.{i}"
        # attn norm (layer 0 may be identity / missing)
        an = f"{p}.attn_norm.weight"
        if an in weights:
            writer.add_tensor(f"blk.{i}.attn_norm.weight", w(an).astype(np.float32))
        # packed QKV
        wqkv = w(f"{p}.attn.Wqkv.weight")
        writer.add_tensor(f"blk.{i}.attn_qkv.weight", wqkv.astype(np.float32))
        wo = w(f"{p}.attn.Wo.weight")
        writer.add_tensor(f"blk.{i}.attn_output.weight", wo.astype(np.float32))
        fn = w(f"{p}.mlp_norm.weight")
        writer.add_tensor(f"blk.{i}.ffn_norm.weight", fn.astype(np.float32))
        # GeGLU: Wi may be fused gate+up
        wi = w(f"{p}.mlp.Wi.weight")
        writer.add_tensor(f"blk.{i}.ffn_up.weight", wi.astype(np.float32))
        wo2 = w(f"{p}.mlp.Wo.weight")
        writer.add_tensor(f"blk.{i}.ffn_down.weight", wo2.astype(np.float32))

    on = w("encoder.final_norm.weight") if "encoder.final_norm.weight" in weights else w(
        "encoder.norm.weight"
    )
    writer.add_tensor("output_norm.weight", on.astype(np.float32))


def _add_head(writer: gguf.GGUFWriter, weights: dict, cfg: dict) -> None:
    n_embd = None
    for k, v in weights.items():
        if k.endswith("type_emb.weight") or k == "type_emb.weight":
            te = v
            # HF layout is [n_types, n_embd] = [3, d]. gguf-py stores numpy
            # dims reversed into ggml ne[], so leave as [3, d] → ne=[d, 3]
            # which matches create_tensor({n_embd, 3}). Do NOT transpose.
            if te.ndim != 2:
                raise ValueError(f"type_emb.weight expected 2D, got {te.shape}")
            n_embd = int(te.shape[1] if te.shape[0] == 3 else te.shape[0])
            writer.add_tensor("laya.type_embd.weight", te.astype(np.float32))
            break
    if n_embd is None:
        raise KeyError("type_emb.weight")

    head_layers = int(cfg.get("head_layers", 2))
    writer.add_uint32("laya.head_layers", head_layers)
    n_act = int(cfg.get("n_act", len(cfg.get("act_costs", {})) + 1))
    writer.add_uint32("laya.n_act", n_act)

    for i in range(head_layers):
        prefix = f"head.layers.{i}"
        # PyTorch TransformerEncoderLayer state_dict keys
        mapping = [
            (f"{prefix}.norm1.weight", f"laya.head.{i}.attn_norm.weight"),
            (f"{prefix}.norm1.bias", f"laya.head.{i}.attn_norm.bias"),
            (f"{prefix}.self_attn.in_proj_weight", f"laya.head.{i}.attn_qkv.weight"),
            (f"{prefix}.self_attn.in_proj_bias", f"laya.head.{i}.attn_qkv.bias"),
            (f"{prefix}.self_attn.out_proj.weight", f"laya.head.{i}.attn_out.weight"),
            (f"{prefix}.self_attn.out_proj.bias", f"laya.head.{i}.attn_out.bias"),
            (f"{prefix}.norm2.weight", f"laya.head.{i}.ffn_norm.weight"),
            (f"{prefix}.norm2.bias", f"laya.head.{i}.ffn_norm.bias"),
            (f"{prefix}.linear1.weight", f"laya.head.{i}.ffn_up.weight"),
            (f"{prefix}.linear1.bias", f"laya.head.{i}.ffn_up.bias"),
            (f"{prefix}.linear2.weight", f"laya.head.{i}.ffn_down.weight"),
            (f"{prefix}.linear2.bias", f"laya.head.{i}.ffn_down.bias"),
        ]
        for src, dst in mapping:
            if src not in weights:
                raise KeyError(src)
            writer.add_tensor(dst, weights[src].astype(np.float32))

    # scorer: Sequential LN, Linear, GELU, Linear → indices 0,1,3
    scorer = [
        ("scorer.0.weight", "laya.scorer.norm.weight"),
        ("scorer.0.bias", "laya.scorer.norm.bias"),
        ("scorer.1.weight", "laya.scorer.fc1.weight"),
        ("scorer.1.bias", "laya.scorer.fc1.bias"),
        ("scorer.3.weight", "laya.scorer.fc2.weight"),
        ("scorer.3.bias", "laya.scorer.fc2.bias"),
    ]
    for src, dst in scorer:
        if src not in weights:
            raise KeyError(src)
        writer.add_tensor(dst, weights[src].astype(np.float32))

    act = [
        ("act_head.0.weight", "laya.act.fc1.weight"),
        ("act_head.0.bias", "laya.act.fc1.bias"),
        ("act_head.2.weight", "laya.act.fc2.weight"),
        ("act_head.2.bias", "laya.act.fc2.bias"),
    ]
    for src, dst in act:
        if src not in weights:
            raise KeyError(src)
        writer.add_tensor(dst, weights[src].astype(np.float32))


def _add_tokenizer_hint(writer: gguf.GGUFWriter, model_dir: Path) -> None:
    tok_json = model_dir / "tokenizer" / "tokenizer.json"
    if tok_json.exists():
        writer.add_string("general.tokenizer_json", str(tok_json))


def _add_tokenizer(writer: gguf.GGUFWriter, model_dir: Path, enc_cfg: dict) -> None:
    """Write GPT-2 BPE vocab (ModernBERT / Laya). WHY required: llama.cpp refuses load without vocab."""
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise SystemExit("pip install transformers — needed to export tokenizer vocab") from e

    tok_dir = model_dir / "tokenizer"
    if not tok_dir.is_dir():
        tok_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), use_fast=True)

    vocab = tokenizer.get_vocab()  # token -> id
    rev = {i: t for t, i in vocab.items()}
    vocab_size = int(enc_cfg.get("vocab_size", max(rev) + 1 if rev else 0))
    if not rev:
        raise SystemExit("empty tokenizer vocab")
    vocab_size = max(vocab_size, max(rev) + 1)

    # Match HF added/special tokens as CONTROL when possible.
    special_ids = set()
    for attr in ("cls_token_id", "sep_token_id", "mask_token_id", "pad_token_id", "unk_token_id", "bos_token_id", "eos_token_id"):
        tid = getattr(tokenizer, attr, None)
        if isinstance(tid, int) and tid >= 0:
            special_ids.add(tid)
    if getattr(tokenizer, "all_special_ids", None):
        special_ids.update(int(x) for x in tokenizer.all_special_ids if isinstance(x, int))

    tokens: list[str] = []
    toktypes: list[int] = []
    unused = int(gguf.TokenType.UNUSED)
    control = int(gguf.TokenType.CONTROL)
    normal = int(gguf.TokenType.NORMAL)
    for i in range(vocab_size):
        if i in rev:
            tokens.append(rev[i])
            toktypes.append(control if i in special_ids else normal)
        else:
            tokens.append(f"[PAD{i}]")
            toktypes.append(unused)

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("modern-bert")
    writer.add_token_list(tokens)
    writer.add_token_types(toktypes)

    # merges + special token ids (cls/sep/mask/…)
    try:
        special = gguf.SpecialVocab(str(tok_dir), load_merges=True)
        special.add_to_gguf(writer)
    except Exception as e:
        print(f"warn: SpecialVocab incomplete ({e}); writing explicit special ids", file=sys.stderr)
        if tokenizer.cls_token_id is not None:
            writer.add_cls_token_id(int(tokenizer.cls_token_id))
        if tokenizer.sep_token_id is not None:
            writer.add_sep_token_id(int(tokenizer.sep_token_id))
        if tokenizer.mask_token_id is not None:
            writer.add_mask_token_id(int(tokenizer.mask_token_id))
        if tokenizer.pad_token_id is not None:
            writer.add_pad_token_id(int(tokenizer.pad_token_id))
        if tokenizer.unk_token_id is not None:
            writer.add_unk_token_id(int(tokenizer.unk_token_id))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True, help="Laya checkpoint directory")
    ap.add_argument("--outfile", type=Path, required=True, help="output .gguf path")
    args = ap.parse_args()

    model_dir = args.model.resolve()
    cfg_path = model_dir / "rl_agent_config.json"
    if not cfg_path.exists():
        raise SystemExit(f"missing {cfg_path}")
    cfg = _load_json(cfg_path)

    enc_dir = model_dir / "encoder"
    enc_cfg_path = enc_dir / "config.json"
    if not enc_cfg_path.exists():
        # Some packs embed encoder config under cfg["encoder"] hub id only.
        raise SystemExit(f"missing {enc_cfg_path} — download encoder config next to weights")
    enc_cfg = _load_json(enc_cfg_path)

    weights = _load_weights(model_dir)

    writer = gguf.GGUFWriter(str(args.outfile), arch="laya")

    # SWA / ModernBERT metadata
    if "sliding_window" in enc_cfg:
        writer.add_uint32("laya.attention.sliding_window", int(enc_cfg["sliding_window"]))
    # WHY laya.rope.freq_base_swa (no ".attention."): matches LLM_KV_ROPE_FREQ_BASE_SWA.
    # Wrong key → silent miss → default SWA RoPE → bad encode on sliding-window layers.
    if "local_rope_theta" in enc_cfg or "global_rope_theta" in enc_cfg:
        writer.add_float32(
            "laya.rope.freq_base_swa",
            float(enc_cfg.get("local_rope_theta", enc_cfg.get("global_rope_theta", 160000))),
        )

    _add_encoder(writer, weights, enc_cfg)
    _add_head(writer, weights, cfg)

    temps = cfg.get("temperature", [1.0, 1.0, 1.0])
    for i, t in enumerate(temps):
        writer.add_float32(f"laya.temperature.{i}", float(t))

    _add_tokenizer(writer, model_dir, enc_cfg)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.outfile}")


if __name__ == "__main__":
    main()
