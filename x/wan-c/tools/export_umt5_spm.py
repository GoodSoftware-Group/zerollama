#!/usr/bin/env python3
"""Export UMT5 SentencePiece .model to wan-c binary vocab format.

Binary layout (little-endian):
  magic u32 0x564f4357 ("WCOV")
  n_pieces u32
  repeat n_pieces:
    score f32
    len u32
    bytes[len]

Usage:
  python3 tools/export_umt5_spm.py /path/to/spiece.model -o umt5.vocab
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

WAN_VOCAB_MAGIC = 0x564F4357


def export_sentencepiece(model_path: Path, out_path: Path) -> None:
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise SystemExit(
            "sentencepiece package required: pip install sentencepiece"
        ) from exc

    proc = spm.SentencePieceProcessor()
    proc.Load(str(model_path))
    n = proc.GetPieceSize()
    pieces: list[tuple[float, bytes]] = []
    for i in range(n):
        piece = proc.IdToPiece(i)
        score = float(proc.GetScore(i))
        pieces.append((score, piece.encode("utf-8")))

    with out_path.open("wb") as f:
        f.write(struct.pack("<I", WAN_VOCAB_MAGIC))
        f.write(struct.pack("<I", n))
        for score, raw in pieces:
            f.write(struct.pack("<fI", score, len(raw)))
            f.write(raw)
    print(f"wrote {out_path} ({n} pieces)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path, help="SentencePiece .model path")
    ap.add_argument("-o", "--out", type=Path, default=Path("umt5.vocab"))
    args = ap.parse_args()
    if not args.model.is_file():
        print(f"missing model: {args.model}", file=sys.stderr)
        raise SystemExit(1)
    export_sentencepiece(args.model, args.out)


if __name__ == "__main__":
    main()
