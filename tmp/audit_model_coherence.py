#!/usr/bin/env python3
"""Coherence audit for local zerollama completion models on :11434."""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HOST = "http://127.0.0.1:11434"
TAGS = Path("/tmp/zl-tags.json")
OUT = Path("/tmp/model-coherence-audit.jsonl")
SUMMARY = Path("/tmp/model-coherence-audit.md")

PROMPT = (
    "Answer with exactly one short English sentence. "
    "What is the capital of France?"
)
EXPECT = re.compile(r"\bparis\b", re.I)
# Soft: readable latin words even if not Paris (template weirdness).
WORD = re.compile(r"[A-Za-z]{3,}")


def http_json(method: str, path: str, body: dict | None = None, timeout: float = 600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        HOST + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode()), None
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            return json.loads(raw), f"HTTP {e.code}"
        except Exception:
            return {"error": raw[:500]}, f"HTTP {e.code}"
    except Exception as e:
        return {"error": str(e)}, str(e)


def pick_ctx(size: int) -> int:
    gib = size / (1024**3)
    if gib >= 70:
        return 1024
    if gib >= 40:
        return 2048
    if gib >= 15:
        return 2048
    return 2048


def local_models():
    d = json.loads(TAGS.read_text())
    out = []
    for m in d.get("models", []):
        name = m.get("name") or ""
        caps = m.get("capabilities") or []
        det = m.get("details") or {}
        if det.get("family") == "cloud" or name.endswith(":cloud") or "-cloud" in name:
            continue
        if "lab-smoke" in name:
            continue
        if "embedding" in caps and "completion" not in caps:
            continue
        if m.get("size", 0) < 1_000_000:
            continue
        if caps and "completion" not in caps:
            continue
        out.append(m)
    # Small → large so we get signal before MoEs.
    out.sort(key=lambda m: m.get("size", 0))
    return out


def unload(name: str):
    http_json(
        "POST",
        "/api/chat",
        {
            "model": name,
            "messages": [{"role": "user", "content": "."}],
            "stream": False,
            "keep_alive": 0,
            "options": {"num_predict": 1, "num_ctx": 256},
        },
        timeout=120,
    )


def score_text(text: str) -> tuple[str, list[str]]:
    """Return (verdict, reasons). verdict: ok|empty|gibberish|weak|wrong."""
    reasons: list[str] = []
    t = (text or "").strip()
    if not t:
        return "empty", ["empty response"]

    # Strip common think wrappers if leaked into content.
    cleaned = re.sub(r"<think>.*?</think>", " ", t, flags=re.I | re.S)
    cleaned = cleaned.strip() or t

    printable = sum(1 for c in cleaned if c.isprintable() or c.isspace())
    if printable / max(len(cleaned), 1) < 0.85:
        reasons.append("low printable ratio")

    words = WORD.findall(cleaned)
    if len(words) < 2:
        reasons.append("too few latin words")

    # Repeated char / token spam
    if re.search(r"(.)\1{12,}", cleaned) or re.search(r"(.{2,8})\1{6,}", cleaned):
        reasons.append("repetition spam")

    # High non-ascii symbol density (not normal punctuation)
    weird = sum(1 for c in cleaned if ord(c) > 127 and not c.isalpha())
    if weird / max(len(cleaned), 1) > 0.25:
        reasons.append("high weird-symbol density")

    if reasons:
        return "gibberish", reasons

    if EXPECT.search(cleaned):
        return "ok", ["mentions Paris"]

    # Coherent English but missed the fact — still not broken weights.
    if len(words) >= 3 and printable / len(cleaned) > 0.95:
        return "weak", ["readable but no Paris"]

    return "wrong", ["unrecognized output"]


def extract_reply(d: dict) -> str:
    if d.get("response"):
        return d["response"]
    msg = d.get("message") or {}
    parts = []
    if msg.get("content"):
        parts.append(msg["content"])
    # Some runners put final answer only after thinking; content should hold it.
    return "\n".join(parts).strip()


def audit_one(m: dict) -> dict:
    name = m["name"]
    size = m.get("size", 0)
    ctx = pick_ctx(size)
    t0 = time.time()
    body = {
        "model": name,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": False,
        "keep_alive": "0",
        "options": {
            "temperature": 0,
            "num_predict": 48,
            "num_ctx": ctx,
        },
    }
    d, err = http_json("POST", "/api/chat", body, timeout=600)
    elapsed = time.time() - t0
    api_err = d.get("error") if isinstance(d, dict) else None
    text = extract_reply(d) if isinstance(d, dict) else ""
    if api_err or err:
        verdict, reasons = "error", [str(api_err or err)]
    else:
        verdict, reasons = score_text(text)
    ec = d.get("eval_count") if isinstance(d, dict) else None
    ed = d.get("eval_duration") if isinstance(d, dict) else None
    tps = None
    if ec and ed:
        tps = round(ec / (ed / 1e9), 1)
    return {
        "name": name,
        "size_gib": round(size / 1024**3, 2),
        "num_ctx": ctx,
        "verdict": verdict,
        "reasons": reasons,
        "response": text[:240],
        "eval_count": ec,
        "tok_s": tps,
        "elapsed_s": round(elapsed, 1),
        "caps": m.get("capabilities"),
    }


def main():
    models = local_models()
    print(f"auditing {len(models)} local models → {OUT}", flush=True)
    OUT.write_text("")
    rows = []
    for i, m in enumerate(models, 1):
        print(f"[{i}/{len(models)}] {m['name']} …", flush=True)
        row = audit_one(m)
        rows.append(row)
        with OUT.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"  → {row['verdict']}  tok/s={row['tok_s']}  "
            f"{row['elapsed_s']}s  {row['response']!r}"[:200],
            flush=True,
        )
        # Best-effort unload (keep_alive=0 already requested).
        try:
            unload(m["name"])
        except Exception:
            pass
        time.sleep(0.5)

    order = ["ok", "weak", "wrong", "empty", "gibberish", "error"]
    lines = ["# Local model coherence audit", "", f"Prompt: `{PROMPT}`", ""]
    for v in order:
        group = [r for r in rows if r["verdict"] == v]
        if not group:
            continue
        lines.append(f"## {v} ({len(group)})")
        lines.append("")
        for r in group:
            snippet = (r["response"] or "")[:120].replace("`", "'")
            extra = ""
            if r["reasons"] and v != "ok":
                extra = " _" + ", ".join(r["reasons"]) + "_"
            lines.append(
                f"- **{r['name']}** — {r['tok_s']} tok/s — `{snippet}`{extra}"
            )
        lines.append("")
    SUMMARY.write_text("\n".join(lines) + "\n")
    counts = {v: sum(1 for r in rows if r["verdict"] == v) for v in order}
    print("DONE", counts, "summary", SUMMARY, flush=True)
    return 0 if counts.get("gibberish", 0) == 0 and counts.get("error", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
