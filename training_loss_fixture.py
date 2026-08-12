"""Fixed-corpus loss-curve fixture for T8 (training packing / padding_free).

This is a **training** check (not inference): prove that:

1. Seeded micro-train losses are deterministic on a fixed fixture.
2. With ``batch_size=1`` and ``packing=false``, ``padding_free`` (flattening)
   matches longest-pad loss (no pads → same tokens / attention).
3. ``packing=true`` is deterministic on the fixture; curves may **diverge**
   from unpacked (concat packing is not Unsloth document-isolated attention).

Used by ``tests/test_training_loss_fixture.py`` and
``scripts/training/loss_curve_fixture.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Repo-root imports when run as script / unittest
_REPO = Path(__file__).resolve().parent


def corpus_path() -> Path:
    return _REPO / "tests" / "fixtures" / "sft_loss_corpus.json"


def load_corpus(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or corpus_path()
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def corpus_texts(data: Optional[Dict[str, Any]] = None) -> Tuple[List[str], int]:
    """Stable fixture strings (prefer ``text``; else alpaca instruction/output)."""
    data = data or load_corpus()
    texts: List[str] = []
    for row in data["samples"]:
        if "text" in row:
            texts.append(str(row["text"]))
            continue
        inst = str(row.get("instruction", "")).strip()
        inp = str(row.get("input", "")).strip()
        out = str(row.get("output", "")).strip()
        if inp:
            prompt = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{inst}\n\n### Response:\n"
        texts.append(prompt + out)
    return texts, int(data.get("max_length", 32))


class _FixtureTokenizer:
    """Tiny char→id tokenizer (no HF download). Vocab 0..127; pad=0, eos=1."""

    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    padding_side = "right"
    model_max_length = 512

    def __call__(
        self,
        text: str,
        *,
        truncation: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        return_tensors: Any = None,
        **kwargs: Any,
    ) -> Dict[str, List[int]]:
        # Map printable-ish bytes into 3..127; keep specials free.
        ids = [3 + (ord(c) % 125) for c in text]
        if truncation and max_length is not None:
            ids = ids[: max(1, int(max_length) - 1)]
        if self.eos_token_id is not None and (not ids or ids[-1] != self.eos_token_id):
            ids = ids + [self.eos_token_id]
            if max_length is not None and len(ids) > max_length:
                ids = ids[:max_length]
                ids[-1] = self.eos_token_id
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def pad(
        self,
        encoded_inputs: Any,
        *,
        return_tensors: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        import torch

        # DataCollatorForLanguageModeling passes a list of feature dicts.
        if isinstance(encoded_inputs, list):
            rows = [list(f["input_ids"]) for f in encoded_inputs]
            masks = [
                list(f.get("attention_mask", [1] * len(f["input_ids"])))
                for f in encoded_inputs
            ]
        else:
            rows = [list(r) for r in encoded_inputs["input_ids"]]
            masks = [
                list(m)
                for m in encoded_inputs.get(
                    "attention_mask", [[1] * len(r) for r in rows]
                )
            ]
        max_len = max(len(r) for r in rows)
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        for row, mask in zip(rows, masks):
            pad_n = max_len - len(row)
            input_ids.append(row + [self.pad_token_id] * pad_n)
            attention_mask.append(list(mask) + [0] * pad_n)
        out: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if return_tensors == "pt":
            out = {k: torch.tensor(v) for k, v in out.items()}
        return out


def make_tiny_model(seed: int = 0, *, n_positions: int = 64):
    """Random tiny GPT-2 for CPU micro-train (no weights download)."""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=128,
        n_positions=n_positions,
        n_embd=32,
        n_layer=2,
        n_head=2,
        n_inner=64,
        bos_token_id=2,
        eos_token_id=1,
        pad_token_id=0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model, _FixtureTokenizer()


def _tokenize_rows(
    texts: Sequence[str],
    tokenizer: Any,
    max_length: int,
    *,
    packing: bool,
) -> List[Dict[str, List[int]]]:
    if packing:
        from training_pack import tokenize_and_pack

        packed = tokenize_and_pack(texts, tokenizer, max_length)
        return [
            {
                "input_ids": packed["input_ids"][i],
                "attention_mask": packed["attention_mask"][i],
                "labels": packed["labels"][i],
            }
            for i in range(len(packed["input_ids"]))
        ]
    rows: List[Dict[str, List[int]]] = []
    for text in texts:
        enc = tokenizer(text, truncation=True, max_length=max_length)
        ids = list(enc["input_ids"])
        rows.append(
            {
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "labels": list(ids),
            }
        )
    return rows


def run_loss_curve(
    *,
    packing: bool = False,
    padding_free: bool = True,
    steps: int = 4,
    batch_size: int = 1,
    seed: int = 0,
    lr: float = 1e-3,
    corpus: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a short SGD loop; return losses + metadata.

    Uses the fixed fixture corpus unless ``corpus`` is provided.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset

    from training_collate import build_sft_collator
    from training_pack import packing_stats

    texts, max_length = corpus_texts(corpus)
    model, tokenizer = make_tiny_model(seed, n_positions=max(64, max_length + 8))
    rows = _tokenize_rows(texts, tokenizer, max_length, packing=packing)
    stats = packing_stats(len(texts), {"input_ids": [r["input_ids"] for r in rows]}) if packing else {
        "samples_in": len(texts),
        "blocks_out": len(rows),
        "pack_ratio": 1.0,
    }

    class _DS(Dataset):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, idx: int) -> Dict[str, List[int]]:
            return rows[idx]

    collator, collate_mode = build_sft_collator(tokenizer, padding_free=padding_free)
    loader = DataLoader(
        _DS(),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    torch.manual_seed(seed)

    losses: List[float] = []
    step = 0
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
            # GPT-2: drop flash-attn-only keys if present
            forward_keys = {"input_ids", "attention_mask", "labels", "position_ids"}
            batch = {k: v for k, v in batch.items() if k in forward_keys}
            out = model(**batch)
            loss = out.loss
            if loss is None:
                raise RuntimeError("model returned no loss; labels missing?")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            step += 1

    return {
        "packing": packing,
        "padding_free": padding_free,
        "collate": collate_mode,
        "batch_size": batch_size,
        "steps": steps,
        "seed": seed,
        "max_length": max_length,
        "n_rows": len(rows),
        "packing_stats": stats,
        "losses": losses,
    }


def assert_curves_close(
    a: Sequence[float],
    b: Sequence[float],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    label: str = "curves",
) -> None:
    if len(a) != len(b):
        raise AssertionError(f"{label}: length {len(a)} != {len(b)}")
    for i, (x, y) in enumerate(zip(a, b)):
        if abs(x - y) > atol + rtol * max(abs(x), abs(y)):
            raise AssertionError(
                f"{label}: step {i}: {x!r} vs {y!r} (atol={atol}, rtol={rtol})"
            )


def dump_curves(path: Path, curves: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(curves, f, indent=2)
        f.write("\n")


def main() -> int:
    """CLI: write loss curves for baseline / padding_free / packing variants."""
    import argparse

    p = argparse.ArgumentParser(description="T8 training loss-curve fixture")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out",
        type=Path,
        default=_REPO / "tests" / "fixtures" / ".sft_loss_curves.json",
    )
    args = p.parse_args()

    variants = [
        {"packing": False, "padding_free": False, "batch_size": 1, "name": "baseline_longest_bs1"},
        {"packing": False, "padding_free": True, "batch_size": 1, "name": "padding_free_bs1"},
        {"packing": False, "padding_free": False, "batch_size": 2, "name": "baseline_longest_bs2"},
        {"packing": False, "padding_free": True, "batch_size": 2, "name": "padding_free_bs2"},
        {"packing": True, "padding_free": True, "batch_size": 1, "name": "packing_on_bs1"},
    ]
    out: Dict[str, Any] = {"seed": args.seed, "steps": args.steps, "curves": {}}
    for v in variants:
        name = v.pop("name")
        result = run_loss_curve(steps=args.steps, seed=args.seed, **v)
        out["curves"][name] = result
        print(f"{name}: {result['losses']}")

    # Soft equality check for bs1 padding_free vs longest
    try:
        assert_curves_close(
            out["curves"]["baseline_longest_bs1"]["losses"],
            out["curves"]["padding_free_bs1"]["losses"],
            label="bs1 padding_free vs longest",
        )
        out["bs1_match"] = True
        print("OK: batch_size=1 padding_free matches longest")
    except AssertionError as e:
        out["bs1_match"] = False
        out["bs1_match_error"] = str(e)
        print(f"FAIL: {e}")

    dump_curves(args.out, out)
    print(f"Wrote {args.out}")
    return 0 if out.get("bs1_match") else 1


if __name__ == "__main__":
    # Allow `python training_loss_fixture.py` from repo root
    if str(_REPO) not in os.sys.path:
        os.sys.path.insert(0, str(_REPO))
    raise SystemExit(main())
