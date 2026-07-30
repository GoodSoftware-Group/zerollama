#!/usr/bin/env python3
"""Render docs/assets/demo-operator-cli.gif for the README.

Terminal-style demo of zerollama ls / ps fields + turn-1 vs turn-2 prompt cache.
Default tables are compact (readable in the frame). Optional --from-live pulls
from :11434 (read-only) and reformats; does not load/unload models.

Usage:
  python3 scripts/marketing/make_readme_demo_gif.py
  python3 scripts/marketing/make_readme_demo_gif.py --from-live
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "demo-operator-cli.gif"

W, H = 1120, 640
BG = (18, 20, 24)
FG = (220, 224, 230)
DIM = (120, 128, 140)
ACCENT = (90, 200, 150)
WARN = (230, 170, 90)
BAR_COLD = (90, 110, 130)
BAR_HOT = (90, 200, 150)
PROMPT = (140, 180, 220)

# Compact curated rows (fit the frame). Live mode refreshes names/PERF when possible.
DEFAULT_LS = [
    "NAME                      SIZE     PARAMS                 PERF",
    "qwen3-coder-next:6bit     64 GB    15.0B MoE 512x10       --",
    "gpt-oss-120b:mxfp4-q8     63 GB    14.9B/979.87M active   --",
    "ornith-35b-optiq:latest   22 GB    34.0B MoE 256x8        54.2",
    "granite4.1:3b-mlx         1.8 GB   425.54M                112.7",
    "llama3.2:3b-mlx           1.8 GB   401.75M                148.9",
    "qwen3:0.6b-mlx            349 MB   74.56M                 222.5",
]

DEFAULT_PS = [
    "NAME                    PROJECT                         SESSION                    PROC",
    "ornith-35b-optiq        hermes-lean/discord:dm:…        hermes:agent:main:dm:…     100% GPU",
    "qwen3.6:35b-a3b-mlx     (background)                    bg:digest:6270b447…        100% GPU",
]


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Monaco.ttf",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def human_bytes(n: int) -> str:
    for unit, div in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= div:
            v = n / div
            return f"{v:.0f} {unit}" if v >= 10 else f"{v:.1f} {unit}"
    return f"{n} B"


def truncate(s: str, n: int) -> str:
    s = (s or "").replace("\t", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=8) as r:
        return json.load(r)


def compact_ls_from_cli(text: str) -> list[str]:
    """Parse `zerollama ls` text into a frame-friendly table."""
    rows: list[str] = ["NAME                      SIZE     PARAMS                 PERF"]
    for ln in text.splitlines()[1:]:
        parts = ln.split()
        if len(parts) < 6:
            continue
        # NAME ID SIZE UNIT PARAMS… PERF MODIFIED…
        name, _id = parts[0], parts[1]
        # size is two tokens usually ("64", "GB")
        size = f"{parts[2]} {parts[3]}"
        # PERF is a float or -- just before humanized modified ("4", "minutes", "ago")
        # Walk from end: "ago" / "Never" / "hour(s)" patterns
        perf = "--"
        params_tokens: list[str]
        if parts[-1] == "ago" and len(parts) >= 8:
            # … PERF N unit ago
            perf = parts[-4]
            params_tokens = parts[4:-4]
        elif parts[-1] == "Never":
            perf = parts[-2]
            params_tokens = parts[4:-2]
        else:
            params_tokens = parts[4:]
        params = " ".join(params_tokens)
        rows.append(
            f"{truncate(name, 24):<24}  {size:<7}  {truncate(params, 22):<22}  {perf}"
        )
        if len(rows) >= 8:
            break
    return rows if len(rows) > 1 else DEFAULT_LS


def compact_ps_from_cli(text: str) -> list[str]:
    """Parse `zerollama ps` into a frame-friendly PROJECT/SESSION table."""
    lines_in = [ln for ln in text.splitlines() if ln.strip()]
    if not lines_in:
        return DEFAULT_PS
    header = lines_in[0]
    show_projects = "PROJECT" in header
    rows = ["NAME                    PROJECT                         SESSION                    PROC"]
    for ln in lines_in[1:]:
        # Columns are padded; split on 2+ spaces.
        cols = [c.strip() for c in re.split(r"\s{2,}", ln.strip()) if c.strip()]
        if not cols:
            continue
        name = truncate(cols[0], 22)
        if show_projects and len(cols) >= 6:
            # NAME PROJECT SESSION ID SIZE PROCESSOR CONTEXT UNTIL
            project = truncate(cols[1], 29) if cols[1] else "(background)"
            session = truncate(cols[2], 24)
            proc = cols[5] if len(cols) > 5 else ""
        elif len(cols) >= 4:
            project = "(no project meta)"
            session = ""
            proc = cols[3] if len(cols) > 3 else ""
        else:
            continue
        rows.append(f"{name:<22}  {project:<29}  {session:<24}  {proc}")
    return rows if len(rows) > 1 else DEFAULT_PS


def live_ls_lines() -> list[str]:
    import subprocess

    bin_path = ROOT / "zerollama"
    cmd = [str(bin_path) if bin_path.exists() else "zerollama", "ls"]
    out = subprocess.check_output(cmd, text=True, timeout=30)
    return compact_ls_from_cli(out)


def live_ps_lines(_host: str = "") -> list[str]:
    import subprocess

    bin_path = ROOT / "zerollama"
    cmd = [str(bin_path) if bin_path.exists() else "zerollama", "ps"]
    out = subprocess.check_output(cmd, text=True, timeout=30)
    return compact_ps_from_cli(out)

def new_frame() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", (W, H), BG)
    return im, ImageDraw.Draw(im)


def draw_chrome(d: ImageDraw.ImageDraw, title: str, f_sm: ImageFont.ImageFont, f_md: ImageFont.ImageFont) -> int:
    d.rounded_rectangle((24, 20, W - 24, H - 20), radius=14, outline=(40, 46, 56), width=2)
    d.ellipse((44, 40, 58, 54), fill=(90, 96, 108))
    d.ellipse((68, 40, 82, 54), fill=(90, 96, 108))
    d.ellipse((92, 40, 106, 54), fill=(90, 96, 108))
    d.text((130, 36), title, fill=DIM, font=f_sm)
    d.text((44, 72), "zerollama — agent megaprompts", fill=FG, font=f_md)
    return 118


def add(frames: list[tuple[Image.Image, int]], im: Image.Image, ms: int) -> None:
    frames.append((im, ms))


def typewriter(
    frames: list[tuple[Image.Image, int]],
    lines: list[str],
    title: str,
    caption: str,
    f_sm: ImageFont.ImageFont,
    f_md: ImageFont.ImageFont,
    f_body: ImageFont.ImageFont,
    hold_ms: int = 2200,
) -> None:
    for visible in range(1, len(lines) + 1):
        im, d = new_frame()
        y = draw_chrome(d, title, f_sm, f_md)
        d.text((44, y), caption, fill=ACCENT, font=f_sm)
        y += 36
        for i, line in enumerate(lines[:visible]):
            color = DIM if i == 0 else FG
            d.text((44, y), truncate(line, 108), fill=color, font=f_body)
            y += 28
        if visible < len(lines):
            d.text((44, y), "▋", fill=PROMPT, font=f_body)
            add(frames, im, 120)
        else:
            add(frames, im, hold_ms)


def bar_scene(
    frames: list[tuple[Image.Image, int]],
    f_sm: ImageFont.ImageFont,
    f_md: ImageFont.ImageFont,
    f_body: ImageFont.ImageFont,
) -> None:
    cold_ms, hot_ms = 390, 95
    max_bar = 760

    def paint(progress_cold: float, progress_hot: float, label: str) -> Image.Image:
        im, d = new_frame()
        y = draw_chrome(d, "demo · prompt cache (L3)", f_sm, f_md)
        d.text((44, y), label, fill=ACCENT, font=f_sm)
        y += 40
        d.text((44, y), "Same megaprompt thread  ·  stable prompt_cache_key", fill=DIM, font=f_sm)
        y += 48
        d.text((44, y), "Turn 1 — stock megaprompt tokenize (1 MiB chat)", fill=FG, font=f_body)
        y += 30
        bw = int(max_bar * progress_cold)
        d.rounded_rectangle((44, y, 44 + max_bar, y + 28), radius=6, fill=(32, 36, 44))
        if bw > 0:
            d.rounded_rectangle((44, y, 44 + bw, y + 28), radius=6, fill=BAR_COLD)
        shown = cold_ms if progress_cold >= 1 else int(cold_ms * progress_cold)
        d.text((44 + max_bar + 16, y + 4), f"{shown} ms", fill=WARN, font=f_body)
        y += 64
        d.text((44, y), "Accelerated tokenize + L3 prefix cache on turn 2+", fill=FG, font=f_body)
        y += 30
        bw2 = int(max_bar * (hot_ms / cold_ms) * progress_hot)
        d.rounded_rectangle((44, y, 44 + max_bar, y + 28), radius=6, fill=(32, 36, 44))
        if bw2 > 0:
            d.rounded_rectangle((44, y, 44 + bw2, y + 28), radius=6, fill=BAR_HOT)
        if progress_hot >= 1:
            hot_label = f"{hot_ms} ms  (~4×)"
        else:
            hot_label = f"{int(hot_ms * progress_hot)} ms"
        d.text((44 + max_bar + 16, y + 4), hot_label, fill=ACCENT, font=f_body)
        y += 70
        d.text((44, y), "Qwen2 chat lab tokenize · L3 skips repeat prefill on same prompt_cache_key", fill=DIM, font=f_sm)
        return im

    for i in range(1, 13):
        add(frames, paint(i / 12, 0.0, "$ agent turn 1 (megaprompt)"), 70)
    add(frames, paint(1.0, 0.0, "$ agent turn 1 (megaprompt)"), 600)
    for i in range(1, 8):
        add(frames, paint(1.0, i / 7, "$ agent turn 2 (same prompt_cache_key)"), 70)
    add(frames, paint(1.0, 1.0, "$ agent turn 2 — next megaprompt is way faster"), 2600)


def title_scene(
    frames: list[tuple[Image.Image, int]],
    f_sm: ImageFont.ImageFont,
    f_md: ImageFont.ImageFont,
    f_lg: ImageFont.ImageFont,
) -> None:
    im, d = new_frame()
    draw_chrome(d, "demo", f_sm, f_md)
    d.text((44, 200), "Built for agent megaprompts", fill=FG, font=f_lg)
    d.text((44, 270), "ls · PARAMS + PERF     ps · PROJECT + SESSION", fill=ACCENT, font=f_md)
    d.text((44, 320), "turn 2 · prompt cache skips repeat prefill", fill=DIM, font=f_md)
    add(frames, im, 1800)


def save_gif(frames: list[tuple[Image.Image, int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Shared adaptive palette from a mid frame for coherent colors.
    base = frames[min(8, len(frames) - 1)][0].convert("P", palette=Image.Palette.ADAPTIVE, colors=64)
    palette = base.getpalette()
    q: list[Image.Image] = []
    durations: list[int] = []
    for im, ms in frames:
        p = im.convert("RGB").quantize(palette=base, dither=Image.Dither.NONE)
        if palette:
            p.putpalette(palette)
        q.append(p)
        durations.append(ms)
    q[0].save(
        path,
        save_all=True,
        append_images=q[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-live", action="store_true", help="Read /api/tags + /api/ps (read-only)")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    f_sm = font(16)
    f_md = font(22)
    f_body = font(16)
    f_lg = font(36)

    ls_lines = DEFAULT_LS
    ps_lines = DEFAULT_PS
    if args.from_live:
        try:
            ls_lines = live_ls_lines()
            ps_lines = live_ps_lines()
            print("live: refreshed ls/ps from ./zerollama CLI")
        except Exception as e:
            print(f"live fetch failed ({e}); using curated defaults")

    frames: list[tuple[Image.Image, int]] = []
    title_scene(frames, f_sm, f_md, f_lg)
    typewriter(
        frames,
        ls_lines,
        "demo · zerollama ls",
        "$ zerollama ls     # PARAMS (MoE/active) + PERF (from bench)",
        f_sm,
        f_md,
        f_body,
        hold_ms=2400,
    )
    typewriter(
        frames,
        ps_lines,
        "demo · zerollama ps",
        "$ zerollama ps     # PROJECT / SESSION — who owns the GPU",
        f_sm,
        f_md,
        f_body,
        hold_ms=2600,
    )
    bar_scene(frames, f_sm, f_md, f_body)

    save_gif(frames, args.out)
    size_kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out} ({size_kb:.0f} KiB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
