#!/usr/bin/env python3
"""Capture turn-1 vs turn-2 prompt_eval timing for the README demo GIF.

Lab ports only by default (11435). Refuses production :11434 unless
--allow-production is set (can contend with Hermes).

Writes JSON consumed by make_readme_demo_gif.py --ttft-json.

  OLLAMA_HOST=127.0.0.1:11435 python3 scripts/marketing/capture_ttft_for_gif.py \\
    --model llama3.2:1b-mlx --out tmp/readme-benches/gif-ttft.json
  python3 scripts/marketing/make_readme_demo_gif.py --from-live \\
    --ttft-json tmp/readme-benches/gif-ttft.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path


def post_chat(base: str, body: dict, timeout: float) -> tuple[dict, float]:
    t0 = time.time()
    req = urllib.request.Request(
        f"{base.rstrip('/')}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    return out, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:11435")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, default=Path("tmp/readme-benches/gif-ttft.json"))
    ap.add_argument("--allow-production", action="store_true")
    ap.add_argument("--num-predict", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=300)
    args = ap.parse_args()

    if "11434" in args.host and not args.allow_production:
        print("refusing :11434 (production). use lab :11435 or --allow-production", file=sys.stderr)
        return 2

    key = f"gif-ttft-{int(time.time())}"
    block = ("The quick brown fox jumps over the lazy dog. " * 16 + "\n") * 30
    system = "You are concise.\n\n" + block
    m1 = [{"role": "system", "content": system}, {"role": "user", "content": "Say hi in three words."}]

    def body(messages: list) -> dict:
        return {
            "model": args.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": args.num_predict,
                "temperature": 0,
                "prompt_cache_key": key,
                "zerollama": {
                    "qos_class": "interactive",
                    "project_id": "demo-gif",
                    "project_name": "readme",
                },
            },
        }

    print(f"turn1 → {args.host} model={args.model}")
    out1, wall1 = post_chat(args.host, body(m1), args.timeout)
    pe1 = (out1.get("prompt_eval_duration") or 0) / 1e6
    asst = (out1.get("message") or {}).get("content") or "hi"
    m2 = m1 + [{"role": "assistant", "content": asst}, {"role": "user", "content": "Say bye in three words."}]
    print("turn2 …")
    out2, wall2 = post_chat(args.host, body(m2), args.timeout)
    pe2 = (out2.get("prompt_eval_duration") or 0) / 1e6

    cached = out2.get("prompt_eval_count")  # not ideal; keep wall + pe
    # Prefer zerollama / usage fields if present
    for k in ("cached_prompt_tokens", "prompt_cached_tokens"):
        if out2.get(k) is not None:
            cached = out2[k]
    metrics = {
        "model": args.model,
        "host": args.host,
        "key": key,
        "turn1_wall_s": wall1,
        "turn2_wall_s": wall2,
        "turn1_prompt_eval_ms": pe1,
        "turn2_prompt_eval_ms": pe2,
        "turn1_prompt_eval_count": out1.get("prompt_eval_count"),
        "turn2_prompt_eval_count": out2.get("prompt_eval_count"),
        "turn2_cached_tokens": cached,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {args.out}")
    if pe2 > 0 and pe1 > 0 and pe2 < pe1:
        print(f"ok: prompt_eval {pe1:.0f}→{pe2:.0f} ms (~{pe1/pe2:.1f}×)")
    else:
        print("note: turn2 prompt_eval not clearly faster — still usable for GIF if wall drops")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
