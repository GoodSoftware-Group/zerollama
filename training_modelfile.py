"""Render Modelfile Go TEMPLATEs for SFT via zerollama (ROADMAP T8).

Serve uses Go ``text/template`` Modelfile TEMPLATEs. Training previously only
had HF Jinja / ChatML / Llama3 / Alpaca approximations. This module shells out
to ``zerollama template render --train`` so train strings match serve.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def find_zerollama_bin() -> Optional[str]:
    env = os.environ.get("ZEROLLAMA_BIN") or os.environ.get("OLLAMA_BIN")
    if env and Path(env).is_file() and os.access(env, os.X_OK):
        return env
    # Prefer repo-local build over a stale PATH install (may lack `template render`).
    for cand in (_repo_root() / "zerollama", _repo_root() / "ollama"):
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    for name in ("zerollama", "ollama"):
        found = shutil.which(name)
        if found:
            return found
    return None


def resolve_modelfile_template(request: Mapping[str, Any]) -> str:
    """Return TEMPLATE text from request fields or a named stock file."""
    raw = request.get("template") or request.get("modelfile_template")
    if isinstance(raw, str) and raw.strip():
        # Inline Go template (contains {{) or path.
        if "{{" in raw:
            return raw
        p = Path(raw).expanduser()
        if p.is_file():
            return p.read_text(encoding="utf-8")
        # Named stock under template/*.gotmpl
        stock = _repo_root() / "template" / f"{raw}.gotmpl"
        if stock.is_file():
            return stock.read_text(encoding="utf-8")
        stock2 = _repo_root() / "template" / raw
        if stock2.is_file():
            return stock2.read_text(encoding="utf-8")
        return raw  # may still be a valid template without {{ (unlikely)

    path = request.get("template_file") or request.get("modelfile_template_file")
    if path:
        return Path(str(path)).expanduser().read_text(encoding="utf-8")

    stock_name = str(request.get("template_name", "chatml")).strip() or "chatml"
    stock = _repo_root() / "template" / f"{stock_name}.gotmpl"
    if not stock.is_file():
        raise FileNotFoundError(
            f"modelfile template not found: set template=... or template_name "
            f"(looked for {stock})"
        )
    return stock.read_text(encoding="utf-8")


def render_modelfile_template(
    template_str: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    train: bool = True,
    bin_path: Optional[str] = None,
) -> str:
    """Render via ``zerollama template render`` (same engine as serve)."""
    bin_path = bin_path or find_zerollama_bin()
    if not bin_path:
        raise RuntimeError(
            "format=modelfile requires a zerollama binary "
            "(build with CGO_ENABLED=1 go build -o zerollama . or set ZEROLLAMA_BIN)"
        )
    payload = {
        "template": template_str,
        "messages": [
            {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in messages
        ],
        "train": bool(train),
    }
    proc = subprocess.run(
        [bin_path, "template", "render"],
        input=json.dumps(payload).encode("utf-8"),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"zerollama template render failed: {err.strip()}")
    return proc.stdout.decode("utf-8")


def format_modelfile(
    sample: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    messages_fn=None,
) -> str:
    """Format one SFT sample with the job's Modelfile TEMPLATE."""
    from training_format import _as_messages  # local cycle-safe late import

    as_messages = messages_fn or _as_messages
    tmpl = resolve_modelfile_template(request)
    return render_modelfile_template(tmpl, as_messages(sample), train=True)
