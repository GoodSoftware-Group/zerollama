#!/usr/bin/env python3
"""Render docs/assets/demo-operator-cli.gif for the README (v2).

Scenes: title → ls (PARAMS/PERF) → ps (PROJECT/SESSION) → harness curl →
tokenize win (measured) → optional live TTFT bars (--ttft-json).

Does not start serve or unload models. --from-live only reads `zerollama ls/ps`.

Usage:
  python3 scripts/marketing/make_readme_demo_gif.py --from-live
  python3 scripts/marketing/make_readme_demo_gif.py --from-live --ttft-json tmp/readme-benches/gif-ttft.json
  # when GPU free (lab port):
  OLLAMA_HOST=127.0.0.1:11435 python3 scripts/marketing/capture_ttft_for_gif.py --model llama3.2:1b-mlx
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "demo-operator-cli.gif"

W, H = 1120, 640
BG = (18, 20, 24)
PANEL = (24, 28, 34)
FG = (220, 224, 230)
DIM = (120, 128, 140)
ACCENT = (90, 200, 150)
WARN = (230, 170, 90)
BAR_COLD = (90, 110, 130)
BAR_HOT = (90, 200, 150)
PROMPT = (140, 180, 220)
HL = (55, 90, 75)  # soft column highlight

# Measured Jul 2026 — docs/readme-marketing-benches.md (Qwen2 chat 1 MiB)
DEFAULT_TOKENIZE = {"cold_ms": 389, "fast_ms": 81, "label": "Qwen2 chat · 1 MiB tokenize", "speedup": "4.8×"}

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
    "qwen3-coder-next        hermes-lean/discord:dm:…        hermes:agent:main:dm:…     100% GPU",
    "qwen3.6:35b-a3b-mlx     (background)                    bg:digest:6270b447…        100% GPU",
]

HARNESS_LINES = [
    "$ curl /api/chat -d '{",
    '  "model": "…",',
    '  "options": {',
    '    "prompt_cache_key": "hermes-thread-42",',
    '    "zerollama": {',
    '      "qos_class": "interactive",',
    '      "project_id": "hermes-lean"',
    "    }",
    "  }",
    "}'",
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


def truncate(s: str, n: int) -> str:
    s = (s or "").replace("\t", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def compact_ls_from_cli(text: str) -> list[str]:
    rows: list[str] = ["NAME                      SIZE     PARAMS                 PERF"]
    for ln in text.splitlines()[1:]:
        parts = ln.split()
        if len(parts) < 6:
            continue
        name = parts[0]
        size = f"{parts[2]} {parts[3]}"
        if parts[-1] == "ago" and len(parts) >= 8:
            perf = parts[-4]
            params_tokens = parts[4:-4]
        elif parts[-1] == "Never":
            perf = parts[-2]
            params_tokens = parts[4:-2]
        else:
            perf = "--"
            params_tokens = parts[4:]
        params = " ".join(params_tokens)
        rows.append(f"{truncate(name, 24):<24}  {size:<7}  {truncate(params, 22):<22}  {perf}")
        if len(rows) >= 8:
            break
    return rows if len(rows) > 1 else DEFAULT_LS


def compact_ps_from_cli(text: str) -> list[str]:
    lines_in = [ln for ln in text.splitlines() if ln.strip()]
    if not lines_in:
        return DEFAULT_PS
    header = lines_in[0]
    show_projects = "PROJECT" in header
    rows = ["NAME                    PROJECT                         SESSION                    PROC"]
    for ln in lines_in[1:]:
        cols = [c.strip() for c in re.split(r"\s{2,}", ln.strip()) if c.strip()]
        if not cols:
            continue
        name = truncate(cols[0], 22)
        if show_projects and len(cols) >= 6:
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
    bin_path = ROOT / "zerollama"
    cmd = [str(bin_path) if bin_path.exists() else "zerollama", "ls"]
    return compact_ls_from_cli(subprocess.check_output(cmd, text=True, timeout=30))


def live_ps_lines() -> list[str]:
    bin_path = ROOT / "zerollama"
    cmd = [str(bin_path) if bin_path.exists() else "zerollama", "ps"]
    return compact_ps_from_cli(subprocess.check_output(cmd, text=True, timeout=30))


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
    frames.append((im, max(20, int(ms))))


def typewriter_table(
    frames: list[tuple[Image.Image, int]],
    lines: list[str],
    title: str,
    caption: str,
    highlight_cols: list[tuple[int, int]],
    f_sm: ImageFont.ImageFont,
    f_md: ImageFont.ImageFont,
    f_body: ImageFont.ImageFont,
    hold_ms: int = 2400,
) -> None:
    for visible in range(1, len(lines) + 1):
        im, d = new_frame()
        y = draw_chrome(d, title, f_sm, f_md)
        d.text((44, y), caption, fill=ACCENT, font=f_sm)
        y += 34
        # column highlight bands behind PARAMS/PERF or PROJECT/SESSION
        if visible >= 1:
            for x0, x1 in highlight_cols:
                d.rectangle((44 + x0, y - 4, 44 + x1, y + 28 * visible + 4), fill=HL)
        for i, line in enumerate(lines[:visible]):
            color = DIM if i == 0 else FG
            d.text((44, y), truncate(line, 108), fill=color, font=f_body)
            y += 28
        if visible < len(lines):
            d.text((44, y), "▋", fill=PROMPT, font=f_body)
            add(frames, im, 110)
        else:
            # badge
            d.rounded_rectangle((W - 220, 36, W - 48, 62), radius=8, fill=(35, 55, 45))
            d.text((W - 208, 40), "operator DX", fill=ACCENT, font=f_sm)
            add(frames, im, hold_ms)


def harness_scene(
    frames: list[tuple[Image.Image, int]],
    f_sm: ImageFont.ImageFont,
    f_md: ImageFont.ImageFont,
    f_body: ImageFont.ImageFont,
) -> None:
    for visible in range(1, len(HARNESS_LINES) + 1):
        im, d = new_frame()
        y = draw_chrome(d, "demo · harness API", f_sm, f_md)
        d.text((44, y), "$ agent turn — stable thread key + QoS", fill=ACCENT, font=f_sm)
        y += 36
        for i, line in enumerate(HARNESS_LINES[:visible]):
            color = ACCENT if "prompt_cache_key" in line or "project_id" in line or "qos_class" in line else FG
            if line.startswith("$"):
                color = DIM
            d.text((44, y), line, fill=color, font=f_body)
            y += 26
        if visible < len(HARNESS_LINES):
            d.text((44, y), "▋", fill=PROMPT, font=f_body)
            add(frames, im, 90)
        else:
            d.text((44, y + 16), "→ turn 2+ reuses prefix (L3) · ps shows PROJECT/SESSION", fill=DIM, font=f_sm)
            add(frames, im, 2200)


def dual_metric_cards(
    frames: list[tuple[Image.Image, int]],
    title: str,
    subtitle: str,
    left_title: str,
    left_value: str,
    left_sub: str,
    right_title: str,
    right_value: str,
    right_sub: str,
    badge: str,
    f_sm: ImageFont.ImageFont,
    f_md: ImageFont.ImageFont,
    f_body: ImageFont.ImageFont,
    f_lg: ImageFont.ImageFont,
    hold_ms: int = 2800,
    animate_right: bool = True,
) -> None:
    """Animate left card then right card filling in."""

    def paint(show_right: float) -> Image.Image:
        im, d = new_frame()
        y = draw_chrome(d, title, f_sm, f_md)
        d.text((44, y), subtitle, fill=ACCENT, font=f_sm)
        y += 44
        # left card
        d.rounded_rectangle((44, y, 540, y + 220), radius=12, fill=PANEL)
        d.text((64, y + 24), left_title, fill=DIM, font=f_sm)
        d.text((64, y + 70), left_value, fill=WARN, font=f_lg)
        d.text((64, y + 150), left_sub, fill=DIM, font=f_body)
        # right card
        d.rounded_rectangle((580, y, W - 44, y + 220), radius=12, fill=PANEL)
        d.text((600, y + 24), right_title, fill=DIM, font=f_sm)
        if show_right > 0:
            # fade value by drawing once fully when show_right>=1
            d.text((600, y + 70), right_value if show_right >= 1 else "…", fill=ACCENT, font=f_lg)
            d.text((600, y + 150), right_sub if show_right >= 1 else "", fill=DIM, font=f_body)
        d.rounded_rectangle((W - 260, 36, W - 48, 62), radius=8, fill=(35, 55, 45))
        d.text((W - 248, 40), badge, fill=ACCENT, font=f_sm)
        return im

    add(frames, paint(0.0), 400)
    if animate_right:
        for i in range(1, 6):
            add(frames, paint(i / 5), 80)
    add(frames, paint(1.0), hold_ms)


def bar_compare(
    frames: list[tuple[Image.Image, int]],
    cold_ms: float,
    hot_ms: float,
    title: str,
    caption: str,
    cold_label: str,
    hot_label: str,
    footer: str,
    badge: str,
    f_sm: ImageFont.ImageFont,
    f_md: ImageFont.ImageFont,
    f_body: ImageFont.ImageFont,
) -> None:
    max_bar = 720
    scale = max(cold_ms, hot_ms, 1.0)

    def paint(pc: float, ph: float, label: str) -> Image.Image:
        im, d = new_frame()
        y = draw_chrome(d, title, f_sm, f_md)
        d.text((44, y), label, fill=ACCENT, font=f_sm)
        y += 36
        d.text((44, y), caption, fill=DIM, font=f_sm)
        y += 40
        d.text((44, y), cold_label, fill=FG, font=f_body)
        y += 28
        bw = int(max_bar * (cold_ms / scale) * pc)
        d.rounded_rectangle((44, y, 44 + max_bar, y + 26), radius=6, fill=(32, 36, 44))
        if bw > 0:
            d.rounded_rectangle((44, y, 44 + bw, y + 26), radius=6, fill=BAR_COLD)
        shown = int(cold_ms * pc) if pc < 1 else int(cold_ms)
        unit = "ms" if cold_ms < 10000 else "ms"
        d.text((44 + max_bar + 14, y + 3), f"{shown} {unit}", fill=WARN, font=f_body)
        y += 56
        d.text((44, y), hot_label, fill=FG, font=f_body)
        y += 28
        bw2 = int(max_bar * (hot_ms / scale) * ph)
        d.rounded_rectangle((44, y, 44 + max_bar, y + 26), radius=6, fill=(32, 36, 44))
        if bw2 > 0:
            d.rounded_rectangle((44, y, 44 + bw2, y + 26), radius=6, fill=BAR_HOT)
        if ph >= 1:
            ratio = cold_ms / hot_ms if hot_ms > 0 else 0
            hot_txt = f"{int(hot_ms)} ms  (~{ratio:.1f}×)"
        else:
            hot_txt = f"{int(hot_ms * ph)} ms"
        d.text((44 + max_bar + 14, y + 3), hot_txt, fill=ACCENT, font=f_body)
        y += 60
        d.text((44, y), footer, fill=DIM, font=f_sm)
        d.rounded_rectangle((W - 280, 36, W - 48, 62), radius=8, fill=(35, 55, 45))
        d.text((W - 268, 40), badge, fill=ACCENT, font=f_sm)
        return im

    for i in range(1, 14):
        add(frames, paint(i / 13, 0.0, "$ turn 1"), 55)
    add(frames, paint(1.0, 0.0, "$ turn 1"), 500)
    for i in range(1, 9):
        add(frames, paint(1.0, i / 8, "$ turn 2"), 55)
    add(frames, paint(1.0, 1.0, "$ turn 2 — faster"), 2600)


def title_scene(
    frames: list[tuple[Image.Image, int]],
    f_sm: ImageFont.ImageFont,
    f_md: ImageFont.ImageFont,
    f_lg: ImageFont.ImageFont,
) -> None:
    im, d = new_frame()
    draw_chrome(d, "demo", f_sm, f_md)
    d.text((44, 180), "Built for agent megaprompts", fill=FG, font=f_lg)
    d.text((44, 250), "ls · PARAMS + PERF", fill=ACCENT, font=f_md)
    d.text((44, 290), "ps · PROJECT + SESSION", fill=ACCENT, font=f_md)
    d.text((44, 340), "harness key · tokenize · turn-2 cache", fill=DIM, font=f_md)
    add(frames, im, 1600)


def end_card(
    frames: list[tuple[Image.Image, int]],
    f_sm: ImageFont.ImageFont,
    f_md: ImageFont.ImageFont,
    f_lg: ImageFont.ImageFont,
) -> None:
    im, d = new_frame()
    draw_chrome(d, "demo", f_sm, f_md)
    d.text((44, 200), "Same Ollama API. Agent control plane.", fill=FG, font=f_lg)
    d.text((44, 280), "distribution: \"zerollama\"", fill=ACCENT, font=f_md)
    d.text((44, 330), "github.com/GoodSoftware-Group/zerollama", fill=DIM, font=f_sm)
    d.text((44, 370), "x.com/spaceodili", fill=DIM, font=f_sm)
    add(frames, im, 2200)


def save_gif(frames: list[tuple[Image.Image, int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = frames[min(10, len(frames) - 1)][0].convert("P", palette=Image.Palette.ADAPTIVE, colors=64)
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


def load_ttft(path: Path | None) -> dict | None:
    if not path or not path.exists():
        return None
    data = json.loads(path.read_text())
    # Accept capture_ttft_for_gif.py shape
    t1 = data.get("turn1_prompt_eval_ms") or data.get("turn1_wall_ms") or data.get("cold_ms")
    t2 = data.get("turn2_prompt_eval_ms") or data.get("turn2_wall_ms") or data.get("hot_ms")
    if t1 is None or t2 is None:
        # wall seconds
        if "turn1_wall_s" in data and "turn2_wall_s" in data:
            t1 = data["turn1_wall_s"] * 1000
            t2 = data["turn2_wall_s"] * 1000
        else:
            return None
    return {
        "cold_ms": float(t1),
        "hot_ms": float(t2),
        "model": data.get("model", "live"),
        "cached": data.get("turn2_cached_tokens") or data.get("cached_tokens"),
        "source": str(path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-live", action="store_true")
    ap.add_argument("--ttft-json", type=Path, default=None, help="Live turn1/turn2 metrics JSON")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    f_sm = font(16)
    f_md = font(22)
    f_body = font(16)
    f_lg = font(34)

    ls_lines = DEFAULT_LS
    ps_lines = DEFAULT_PS
    if args.from_live:
        try:
            ls_lines = live_ls_lines()
            ps_lines = live_ps_lines()
            print("live: refreshed ls/ps from ./zerollama CLI")
        except Exception as e:
            print(f"live fetch failed ({e}); using curated defaults")

    tok = DEFAULT_TOKENIZE
    ttft = load_ttft(args.ttft_json)
    if ttft:
        print(f"ttft: {ttft['cold_ms']:.0f}→{ttft['hot_ms']:.0f} ms from {ttft['source']}")

    frames: list[tuple[Image.Image, int]] = []
    title_scene(frames, f_sm, f_md, f_lg)

    # monospace approx: highlight PARAMS+PERF region
    typewriter_table(
        frames,
        ls_lines,
        "demo · zerollama ls",
        "$ zerollama ls     # PARAMS (MoE/active) + PERF (from bench)",
        highlight_cols=[(320, 620)],
        f_sm=f_sm,
        f_md=f_md,
        f_body=f_body,
        hold_ms=2300,
    )
    typewriter_table(
        frames,
        ps_lines,
        "demo · zerollama ps",
        "$ zerollama ps     # PROJECT / SESSION — who owns the GPU",
        highlight_cols=[(200, 780)],
        f_sm=f_sm,
        f_md=f_md,
        f_body=f_body,
        hold_ms=2500,
    )
    harness_scene(frames, f_sm, f_md, f_body)

    dual_metric_cards(
        frames,
        title="demo · megaprompt tokenize",
        subtitle=tok["label"] + "  ·  measured lab",
        left_title="Stock / legacy BPE",
        left_value=f"{tok['cold_ms']} ms",
        left_sub="before any forward pass",
        right_title="Zerollama accelerated",
        right_value=f"{tok['fast_ms']} ms",
        right_sub=f"{tok['speedup']}  ·  Gigatoken-inspired",
        badge="measured",
        f_sm=f_sm,
        f_md=f_md,
        f_body=f_body,
        f_lg=f_lg,
        hold_ms=2600,
    )

    if ttft:
        ratio = ttft["cold_ms"] / ttft["hot_ms"] if ttft["hot_ms"] > 0 else 0
        cached = ttft.get("cached")
        footer = f"model {ttft['model']}"
        if cached is not None:
            footer += f"  ·  turn2 cached_tokens={cached}"
        bar_compare(
            frames,
            cold_ms=ttft["cold_ms"],
            hot_ms=ttft["hot_ms"],
            title="demo · prompt cache (L3)",
            caption="Same prompt_cache_key  ·  live capture",
            cold_label="Turn 1 — cold prefill (prompt_eval)",
            hot_label=f"Turn 2 — prefix hit (~{ratio:.1f}×)" if ratio else "Turn 2 — prefix hit",
            footer=footer,
            badge="live TTFT",
            f_sm=f_sm,
            f_md=f_md,
            f_body=f_body,
        )
    else:
        # Honest placeholder: L3 story without fake TTFT numbers
        dual_metric_cards(
            frames,
            title="demo · prompt cache (L3)",
            subtitle="Same thread key → skip repeat prefill  ·  SGLang/vLLM-inspired",
            left_title="Turn 1",
            left_value="full prefill",
            left_sub="system + tools + history",
            right_title="Turn 2+",
            right_value="prefix hit",
            right_sub="stable prompt_cache_key",
            badge="L3",
            f_sm=f_sm,
            f_md=f_md,
            f_body=f_body,
            f_lg=f_lg,
            hold_ms=2400,
        )

    end_card(frames, f_sm, f_md, f_lg)
    save_gif(frames, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size/1024:.0f} KiB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
