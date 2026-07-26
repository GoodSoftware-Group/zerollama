#!/usr/bin/env python3
"""Unit reference for Dual Chunk Attention helpers (masks / RoPE remaps / s(L)).

Mirrors vendor llama-dca.h — no GPU / SGLang required. Exit non-zero on mismatch.

  python3 scripts/dca_unit_ref.py
"""
from __future__ import annotations

import math
import sys


def chunk_len(chunk_size: int, local_size: int) -> int:
    return chunk_size - local_size


def pos_k(pos: int, c_len: int) -> int:
    return pos % c_len if c_len else pos


def pos_q_intra(pos: int, c_len: int) -> int:
    return pos_k(pos, c_len)


def pos_q_succ(pos: int, c_len: int, chunk_size: int) -> int:
    if not c_len:
        return pos
    local = pos % c_len
    return min(local + c_len, chunk_size)


def pos_q_inter(chunk_size: int) -> int:
    return chunk_size


def length_scale(seq_len: int, l0: int) -> float:
    if l0 == 0 or seq_len == 0:
        return 1.0
    return max(1.0, 0.1 * math.log(seq_len / l0) + 1.0)


def chunk_index(pos: int, c_len: int) -> int:
    return pos // c_len if c_len else 0


def allow_key(kind: str, q: int, k: int, c_len: int) -> bool:
    if not c_len:
        return k <= q
    n = chunk_index(q, c_len)
    kc = k // c_len
    if kind == "intra":
        return kc == n and k <= q
    if kind == "succ":
        return n >= 1 and kc == (n - 1)
    if kind == "inter":
        return n >= 2 and k < (n - 1) * c_len
    raise ValueError(kind)


def main() -> int:
    chunk_size, local_size, l0 = 256, 64, 32768
    c = chunk_len(chunk_size, local_size)  # 192
    assert c == 192

    # RoPE remaps at / past chunk boundary
    assert pos_k(5, c) == 5
    assert pos_k(192, c) == 0
    assert pos_q_intra(200, c) == 8
    assert pos_q_succ(5, c, chunk_size) == min(5 + c, chunk_size)
    assert pos_q_succ(200, c, chunk_size) == min(8 + c, chunk_size)
    assert pos_q_inter(chunk_size) == chunk_size

    # s(L): short seq stays 1; long seq grows
    assert length_scale(5, l0) == 1.0
    assert length_scale(l0, l0) == 1.0
    assert length_scale(2 * l0, l0) > 1.0

    # Masks: n=0 only intra; n=1 gains succ; n=2 gains inter
    q0 = 100  # n=0
    assert allow_key("intra", q0, 50, c)
    assert not allow_key("succ", q0, 50, c)
    assert not allow_key("inter", q0, 50, c)

    q1 = 200  # n=1, chunk_len=192
    assert chunk_index(q1, c) == 1
    assert allow_key("intra", q1, 195, c)
    assert allow_key("succ", q1, 100, c)  # prev chunk
    assert not allow_key("inter", q1, 10, c)

    q2 = 400  # n=2
    assert chunk_index(q2, c) == 2
    assert allow_key("inter", q2, 10, c)
    assert allow_key("succ", q2, 200, c)
    assert not allow_key("succ", q2, 10, c)

    # LSE merge: equal LSEs → equal weights → identity
    import statistics

    def soft_max(xs: list[float]) -> list[float]:
        m = max(xs)
        e = [math.exp(x - m) for x in xs]
        s = sum(e)
        return [v / s for v in e]

    w = soft_max([1.0, 1.0, 1.0])
    assert all(abs(x - 1 / 3) < 1e-9 for x in w)
    w0 = soft_max([0.0, -1e9, -1e9])
    assert abs(w0[0] - 1.0) < 1e-6 and w0[1] < 1e-6 and w0[2] < 1e-6

    print("dca_unit_ref: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
