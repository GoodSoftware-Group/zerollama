#!/usr/bin/env python3
"""Oracle harness: native Dual Chunk Attention vs SGLang dense DCA / stock FA.

Cases (see docs/dca-dual-chunk-attention.md):
  n=0 (S < chunk_len): native DCA logits ≈ stock llama FA (intra only)
  n≥1 / n≥2:           native DCA logits ≈ SGLang dense DCA (same HF weights)

Exit non-zero on drift beyond atol/rtol / max abs logit delta.

Examples:
  # Stock vs native DCA at short context (no SGLang required):
  python3 scripts/dca_oracle_logits.py --gguf /path/model-dca.gguf --mode n0

  # Cross-check vs SGLang dense:
  python3 scripts/dca_oracle_logits.py --hf /path/Qwen2.5-...-1M \\
      --gguf /path/model-dca.gguf --mode n1 --sglang-url http://127.0.0.1:30000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.request
from typing import Any, Optional


def _post_json(url: str, payload: dict[str, Any], timeout: float = 600.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def llama_server_logits(
    base: str,
    prompt: str,
    n_predict: int = 1,
    temperature: float = 0.0,
) -> list[float]:
    """Fetch next-token logits from llama-server (completion + logits)."""
    # Prefer /completion with n_probs / logits if available; fall back to embedding-free greedy.
    url = base.rstrip("/") + "/completion"
    body = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
        "cache_prompt": True,
        "n_probs": 0,
        "logits": True,
    }
    try:
        out = _post_json(url, body)
    except Exception as e:
        raise SystemExit(f"llama-server request failed ({url}): {e}") from e

    # llama.cpp server may return logits under different keys across versions
    for key in ("logits", "completion_logits", "scores"):
        if key in out and out[key]:
            logits = out[key]
            if isinstance(logits[0], list):
                logits = logits[-1]
            return [float(x) for x in logits]
    raise SystemExit(
        f"No logits in llama-server response; keys={list(out.keys())}. "
        "Need a build with logits export (or use --compare-greedy only)."
    )


def sglang_logits(base: str, prompt: str) -> list[float]:
    """SGLang OpenAI-compatible: use /v1/completions with logprobs if available."""
    url = base.rstrip("/") + "/v1/completions"
    body = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": 5,
    }
    try:
        out = _post_json(url, body)
    except Exception as e:
        raise SystemExit(f"SGLang request failed ({url}): {e}") from e
    # Prefer raw logits endpoint if operator exposed one
    if "logits" in out:
        return [float(x) for x in out["logits"]]
    raise SystemExit(
        "SGLang response has no raw logits. Launch with a logits dump helper "
        "or compare greedy token ids via --compare-greedy."
    )


def max_abs_delta(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return float("inf")
    return max(abs(a[i] - b[i]) for i in range(n))


def make_prompt(n_tokens_approx: int, pad: str = "hello ") -> str:
    """Build a repetitive prompt aiming for ~n_tokens_approx (whitespace-token heuristic)."""
    unit = pad
    reps = max(1, n_tokens_approx // max(1, len(unit.split())))
    return (unit * reps).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("n0", "n1", "n2"), required=True,
                    help="n0: S<chunk_len; n1: one prior chunk; n2: inter active")
    ap.add_argument("--chunk-len", type=int, default=7168,
                    help="chunk_len = chunk_size - local_size (default Qwen 8192-1024)")
    ap.add_argument("--native-url", default=os.environ.get("DCA_NATIVE_URL", "http://127.0.0.1:8080"),
                    help="llama-server / zerollama with native DCA GGUF")
    ap.add_argument("--stock-url", default=os.environ.get("DCA_STOCK_URL", ""),
                    help="llama-server without DCA (or same binary + dca_*=0 GGUF) for n0")
    ap.add_argument("--sglang-url", default=os.environ.get("DCA_SGLANG_URL", ""),
                    help="SGLang dense dual_chunk_flash_attn (oracle for n≥1)")
    ap.add_argument("--prompt", default="", help="Override prompt text")
    ap.add_argument("--atol", type=float, default=0.05)
    ap.add_argument("--rtol", type=float, default=0.01)
    ap.add_argument("--max-abs", type=float, default=0.5,
                    help="Fail if max |Δlogit| exceeds this")
    ap.add_argument("--compare-greedy", action="store_true",
                    help="Only compare greedy next-token id (when logits unavailable)")
    args = ap.parse_args()

    c = args.chunk_len
    if args.prompt:
        prompt = args.prompt
    elif args.mode == "n0":
        prompt = make_prompt(max(16, c // 8))
    elif args.mode == "n1":
        prompt = make_prompt(c + 64)
    else:
        prompt = make_prompt(2 * c + 64)

    print(f"mode={args.mode} chunk_len={c} prompt_chars={len(prompt)}")

    if args.compare_greedy:
        # Minimal smoke: greedy token string match
        def greedy(url: str) -> str:
            out = _post_json(url.rstrip("/") + "/completion", {
                "prompt": prompt, "n_predict": 1, "temperature": 0.0,
            })
            return out.get("content") or out.get("choices", [{}])[0].get("text", "")

        native = greedy(args.native_url)
        if args.mode == "n0":
            if not args.stock_url:
                print("n0 greedy needs --stock-url", file=sys.stderr)
                return 2
            ref = greedy(args.stock_url)
            label = "stock"
        else:
            if not args.sglang_url:
                print("n≥1 greedy needs --sglang-url", file=sys.stderr)
                return 2
            ref = greedy(args.sglang_url)
            label = "sglang"
        ok = native == ref
        print(f"greedy native={native!r} {label}={ref!r} ok={ok}")
        return 0 if ok else 1

    native_logits = llama_server_logits(args.native_url, prompt)

    if args.mode == "n0":
        if not args.stock_url:
            print("n0 requires --stock-url (stock FA reference)", file=sys.stderr)
            return 2
        ref_logits = llama_server_logits(args.stock_url, prompt)
        label = "stock"
    else:
        if not args.sglang_url:
            print("n≥1 requires --sglang-url (SGLang dense DCA oracle)", file=sys.stderr)
            return 2
        ref_logits = sglang_logits(args.sglang_url, prompt)
        label = "sglang"

    mad = max_abs_delta(native_logits, ref_logits)
    # Relative check on a sample of top-magnitude ref logits
    idxs = sorted(range(len(ref_logits)), key=lambda i: abs(ref_logits[i]), reverse=True)[:64]
    rel_ok = all(
        abs(native_logits[i] - ref_logits[i]) <= args.atol + args.rtol * abs(ref_logits[i])
        for i in idxs if i < len(native_logits)
    )
    abs_ok = mad <= args.max_abs
    print(f"max_abs_delta={mad:.6g} vs {label}; abs_ok={abs_ok} rel_ok={rel_ok}")
    if not (abs_ok and rel_ok):
        print("ORACLE FAIL: logit drift exceeds thresholds", file=sys.stderr)
        return 1
    print("ORACLE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
