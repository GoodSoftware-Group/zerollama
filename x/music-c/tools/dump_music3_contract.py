#!/usr/bin/env python3
"""Dump MiniMax Music 3 prompt/chunk/DAV geometry (Omni contract, CPU, no weights).

Why Omni not Comfy: Apache-2.0 rematch gold; Comfy lyrics normalize is not byte-identical.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SPECIAL_TOKEN_IDS = {
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<|audio_cfg|>": 151654,
    "<|audio_start|>": 151669,
    "<|audio_end|>": 151670,
    "<|caption_start|>": 151671,
    "<|caption_end|>": 151672,
    "<|lyrics_start|>": 151673,
    "<|lyrics_end|>": 151674,
}
AUDIO_CODE_OFFSET = 151675
AR_CHUNK_FRAMES = 200
AR_CHUNK_HOP_FRAMES = 100
AR_HIDDEN_SIZE = 4096 * 8
DAV_SAMPLE_RATE = 44100
OUTPUT_SAMPLE_RATE = 32000
SR_INPUT = 24000
HOP_IN = 960
HOP_OUT = 512
MAX_PROMPT_TOKENS = 5000
MAX_AUDIO_FRAMES = 9000

_SPECIAL_TAG_RE = re.compile(r"<\|([^|]*)\|>")
_LEADING_TAGS_RE = re.compile(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")


def _remove_markdown_format(text: str) -> str:
    lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", raw_line)
        line = re.sub(r"^\s*[*+-]\s+", "", line)
        while "**" in line:
            updated = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            if updated == line:
                break
            line = updated
        line = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)
        lines.append(line.rstrip())
    text = "\n".join(lines)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    return text.replace("• ", "").replace("    ", "")


def clean_caption(caption: str) -> str:
    def _special(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        parts = inner.split(None, 1)
        return f"{parts[0]} is {parts[1]}" if len(parts) == 2 else inner

    text = _SPECIAL_TAG_RE.sub(_special, caption)
    text = _remove_markdown_format(text)
    return re.sub(r"\n{2,}", "\n", text)


def _strip_text_after_leading_tags(text: str) -> str:
    if not text:
        return text
    output = []
    for line in text.split("\n"):
        match = _LEADING_TAGS_RE.match(line)
        output.append(match.group(1).strip() if match else line)
    return "\n".join(output)


def normalize_lyrics(lyrics: str) -> str:
    text = _strip_text_after_leading_tags(lyrics)
    text = text.replace("] ", "]\n")
    text = text.replace(" [", "\n[")
    text = text.replace(" ^ ", "\n")
    text = re.sub(r"\[([^\]]+)\]", lambda m: f"[{m.group(1).lower()}]", text)
    return f"[start]\n{text}"


def build_prompt(caption: str, lyrics: str) -> str:
    return (
        "<|im_start|><|caption_start|>"
        f"{clean_caption(caption)}"
        "<|caption_end|><|lyrics_start|>"
        f"{normalize_lyrics(lyrics)}"
        "<|lyrics_end|><|im_end|><|audio_start|>"
    )


def aligned_mel_length(frames: int) -> int:
    return max(1, int(frames * DAV_SAMPLE_RATE / SR_INPUT * HOP_IN / HOP_OUT))


def chunk_windows(frames: int) -> list[dict[str, int | bool]]:
    frames = int(frames)
    if frames == 0:
        return []
    if frames <= AR_CHUNK_FRAMES:
        return [{"index": 0, "start": 0, "end": frames, "is_first": True, "is_last": True}]
    windows = []
    index = 0
    start = 0
    while start < frames:
        end = min(start + AR_CHUNK_FRAMES, frames)
        windows.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "is_first": index == 0,
                "is_last": end >= frames,
            }
        )
        if end >= frames:
            break
        index += 1
        start += AR_CHUNK_HOP_FRAMES
    return windows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", type=Path, default=Path("x/music-c/fixtures/music3_contract.json"))
    args = ap.parse_args()
    lyrics_same_line = "[Verse] Walking down the street"
    lyrics_own_line = "[Verse]\nWalking down the street"
    payload = {
        "special_token_ids": SPECIAL_TOKEN_IDS,
        "audio_code_offset": AUDIO_CODE_OFFSET,
        "ar_hidden_size": AR_HIDDEN_SIZE,
        "ar_chunk_frames": AR_CHUNK_FRAMES,
        "ar_chunk_hop_frames": AR_CHUNK_HOP_FRAMES,
        "dav_sample_rate": DAV_SAMPLE_RATE,
        "output_sample_rate": OUTPUT_SAMPLE_RATE,
        "dav_upsample": 512,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "max_audio_frames": MAX_AUDIO_FRAMES,
        "aligned_mel_length_250": aligned_mel_length(250),
        "chunk_windows_250": chunk_windows(250),
        "chunk_windows_450": chunk_windows(450),
        "prompt_own_line": build_prompt("Warm acoustic pop", lyrics_own_line),
        "prompt_same_line": build_prompt("Warm acoustic pop", lyrics_same_line),
        "normalized_same_line": normalize_lyrics(lyrics_same_line),
        "normalized_own_line": normalize_lyrics(lyrics_own_line),
    }
    args.o.parent.mkdir(parents=True, exist_ok=True)
    args.o.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.o)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
