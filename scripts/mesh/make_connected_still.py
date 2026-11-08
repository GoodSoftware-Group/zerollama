#!/usr/bin/env python3
"""Write a simple RGBA product still with one connected object (lab input).

The earlier /tmp/trellis-lab/still_mug.png had a floating handle. TRELLIS.2 then
emitted thousands of face-connected components. This still keeps the handle
joined to the body so the image-to-3D path sees one object.

Usage:
  python scripts/mesh/make_connected_still.py --out /tmp/trellis-lab/still_mug_connected.png
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/tmp/trellis-lab/still_mug_connected.png")
    ap.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()

    from PIL import Image, ImageDraw

    size = int(args.size)
    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    d = ImageDraw.Draw(img)

    # Body (centered cylinder-ish rounded rect) + attached C-handle.
    cx, cy = size // 2, size // 2 + 40
    bw, bh = 280, 360
    left, top = cx - bw // 2, cy - bh // 2
    right, bottom = left + bw, top + bh
    red = (200, 40, 40, 255)
    dark = (140, 20, 20, 255)

    d.rounded_rectangle([left, top, right, bottom], radius=40, fill=red)
    # Rim ellipse
    d.ellipse([left + 20, top - 20, right - 20, top + 40], outline=dark, width=10)
    d.ellipse([left + 40, top - 5, right - 40, top + 25], fill=(180, 30, 30, 255))

    # Handle attached to the right side (overlaps body so it is connected).
    hx0 = right - 30
    hx1 = right + 140
    hy0 = top + 80
    hy1 = bottom - 80
    d.arc([hx0, hy0, hx1, hy1], start=270, end=90, fill=red, width=48)
    # Bridge pixels into the body so connectivity is unambiguous in the alpha mask.
    d.rectangle([right - 40, hy0 + 40, right + 10, hy0 + 90], fill=red)
    d.rectangle([right - 40, hy1 - 90, right + 10, hy1 - 40], fill=red)

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"wrote {out} size={size}x{size} mode=RGBA connected_handle=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
