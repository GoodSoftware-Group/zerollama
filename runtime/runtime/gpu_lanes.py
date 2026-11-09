"""Split talk and see across two GPUs.

Voice (llama-server) stays on ``talk_gpu``. Vision ggml stays on ``see_gpu``
(applied in the Go scheduler). A model that does not fit on the talk GPU keeps
the topology tensor split and shares the cards.
"""

from __future__ import annotations

import os
import subprocess

_TS_FLAGS = frozenset({"-ts", "--tensor-split", "--tensor_split"})
_MG_FLAGS = frozenset({"-mg", "--main-gpu", "--main_gpu"})
_NP_FLAGS = frozenset({"-np", "--parallel"})
_CTX_FLAGS = frozenset({"-c", "--ctx-size"})


def lane_index(name: str, fallback: int | None = None) -> int | None:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return fallback
    try:
        n = int(raw)
    except ValueError:
        return fallback
    if n < 0:
        return fallback
    return n


def nvidia_total_vram_bytes(device_index: int) -> int | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = out.strip().splitlines()
    if not line:
        return None
    try:
        mib = int(line[0].strip().split()[0])
    except ValueError:
        return None
    if mib <= 0:
        return None
    return mib * 1024 * 1024


def fits_on_one_gpu(estimated_bytes: int | None, total_bytes: int | None) -> bool:
    if not estimated_bytes or not total_bytes:
        return False
    # Leave a gigabyte for the desktop and CUDA context on that card.
    return estimated_bytes + (1 << 30) <= total_bytes


def _flag_value(args: list[str], names: frozenset[str]) -> int | None:
    for i, flag in enumerate(args):
        if flag in names and i + 1 < len(args):
            try:
                return int(args[i + 1])
            except ValueError:
                return None
    return None


def _replace_flag(args: list[str], names: frozenset[str], value: str) -> list[str]:
    out: list[str] = []
    found = False
    i = 0
    while i < len(args):
        if args[i] in names and i + 1 < len(args):
            out.extend([args[i], value])
            found = True
            i += 2
            continue
        out.append(args[i])
        i += 1
    if not found:
        out.extend([next(iter(names)), value])
    return out


def talk_lane_candidates(args: list[str]) -> list[list[str]]:
    """One slot, then a smaller context if that still does not fit.

    Duplex is a single speaker. Extra slots reserve KV the reply never uses,
    fill the card, and the next turn then clamps context down to a few
    hundred tokens. 8k is the floor because duplex replies fit inside it.
    """
    one_slot = _replace_flag(args, _NP_FLAGS, "1")
    out = [one_slot]
    ctx = _flag_value(one_slot, _CTX_FLAGS)
    if ctx is not None and ctx > 8192:
        for smaller in (16384, 8192):
            if smaller < ctx:
                shrunk = _replace_flag(one_slot, _CTX_FLAGS, str(smaller))
                if shrunk != out[-1]:
                    out.append(shrunk)
    return out


def pin_talk_lane(args: list[str], *, fits: bool, talk_gpu: int) -> tuple[list[str], dict[str, str]]:
    """Drop tensor split and hide every GPU except the talk card."""
    if not fits:
        return list(args), {}
    out: list[str] = []
    i = 0
    while i < len(args):
        flag = args[i]
        if flag in _TS_FLAGS and i + 1 < len(args):
            i += 2
            continue
        if flag in _MG_FLAGS and i + 1 < len(args):
            # The child only sees one device, so its main GPU is ordinal 0.
            out.extend([flag, "0"])
            i += 2
            continue
        out.append(flag)
        i += 1
    # The child sees one device. A parent LLAMA_ARG_DEVICE=CUDA1 would look
    # for a second card that this process cannot see.
    return out, {
        "CUDA_VISIBLE_DEVICES": str(talk_gpu),
        "LLAMA_ARG_DEVICE": "CUDA0",
    }


def plan_talk_lane(
    args: list[str],
    *,
    talk_gpu: int | None,
    estimated_bytes: int | None,
    total_bytes: int | None = None,
) -> tuple[list[str], dict[str, str]]:
    if talk_gpu is None:
        return list(args), {}
    total = total_bytes if total_bytes is not None else nvidia_total_vram_bytes(talk_gpu)
    return pin_talk_lane(args, fits=fits_on_one_gpu(estimated_bytes, total), talk_gpu=talk_gpu)
