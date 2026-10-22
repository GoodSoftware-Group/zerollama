#!/usr/bin/env python3
"""T11 — lite data recipes: files → ``[{prompt, response}, …]`` for ``/api/train``.

WHY: operators need spreadsheet / chat-log → SFT rows without Unsloth Studio (AGPL)
or a second training product. Offline ingest only — they POST to ``/api/train`` themselves
(choose host, T6 window, priority).

stdlib-only for CSV / JSONL / paired TXT. PDF/DOCX need optional deps with clear errors
(WHY optional: do not block CSV users on OCR stacks).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


Row = Dict[str, str]

# T11c chat-export presets (JSON/JSONL objects). WHY named presets: ChatGPT/ShareGPT
# dumps do not use prompt/response keys; operators should not hand-rewrite exports.
_PRESETS = frozenset({"auto", "sharegpt", "openai", "alpaca", "prompt_response"})


def _clean(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()


def rows_to_training_data(rows: Iterable[Row]) -> List[Row]:
    out: List[Row] = []
    for r in rows:
        p, resp = _clean(r.get("prompt")), _clean(r.get("response"))
        if p and resp:
            out.append({"prompt": p, "response": resp})
    return out


def detect_format(path: Path, fmt: Optional[str] = None) -> str:
    if fmt and fmt != "auto":
        return fmt.lower().strip()
    suf = path.suffix.lower()
    if suf in (".csv", ".tsv"):
        return "csv" if suf == ".csv" else "tsv"
    if suf in (".jsonl", ".ndjson"):
        return "jsonl"
    if suf == ".json":
        return "json"
    if suf in (".txt", ".md"):
        return "txt"
    if suf == ".pdf":
        return "pdf"
    if suf in (".docx",):
        return "docx"
    raise ValueError(f"cannot detect format for {path} (pass --format)")


def normalize_preset(preset: Optional[str]) -> str:
    if not preset or not str(preset).strip():
        return "auto"
    p = str(preset).strip().lower().replace("-", "_")
    aliases = {
        "share_gpt": "sharegpt",
        "vicuna": "sharegpt",
        "chatgpt": "openai",
        "chat": "openai",
        "messages": "openai",
        "sft": "prompt_response",
        "pr": "prompt_response",
    }
    p = aliases.get(p, p)
    if p not in _PRESETS:
        raise ValueError(
            f"unknown preset {preset!r} "
            "(want auto|sharegpt|openai|alpaca|prompt_response)"
        )
    return p


def _pair_from_messages(messages: Any) -> Tuple[str, str]:
    """Last user + following assistant (OpenAI chat shape)."""
    if not isinstance(messages, list):
        return "", ""
    user, asst = "", ""
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = _clean(m.get("role")).lower()
        content = m.get("content")
        if isinstance(content, list):
            # Multimodal content parts — take text only.
            parts = [
                _clean(p.get("text"))
                for p in content
                if isinstance(p, dict) and p.get("type") in (None, "text")
            ]
            content = "\n".join(x for x in parts if x)
        content = _clean(content)
        if role == "user" and content:
            user = content
            asst = ""  # reset so we pair with the next assistant
        elif role in ("assistant", "model") and content:
            asst = content
    return user, asst


def _pair_from_sharegpt(obj: Dict[str, Any]) -> Tuple[str, str]:
    conv = obj.get("conversations") or obj.get("conversation")
    if not isinstance(conv, list):
        return "", ""
    user, asst = "", ""
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        frm = _clean(turn.get("from") or turn.get("role")).lower()
        val = _clean(turn.get("value") or turn.get("content") or turn.get("text"))
        if not val:
            continue
        if frm in ("human", "user", "man"):
            user = val
            asst = ""
        elif frm in ("gpt", "assistant", "bot", "chatgpt", "model"):
            asst = val
        # system / observation: ignore for prompt/response pair
    return user, asst


def _pair_from_alpaca(obj: Dict[str, Any]) -> Tuple[str, str]:
    instr = _clean(obj.get("instruction") or obj.get("instruct"))
    inp = _clean(obj.get("input") or obj.get("context"))
    out = _clean(obj.get("output") or obj.get("response") or obj.get("answer"))
    if not instr and not inp:
        return "", out
    if inp:
        prompt = f"{instr}\n\n{inp}" if instr else inp
    else:
        prompt = instr
    return prompt, out


def pair_from_obj(obj: Dict[str, Any], *, preset: str = "auto") -> Tuple[str, str]:
    """Extract (prompt, response) from one JSON object using a chat-export preset."""
    preset = normalize_preset(preset)
    if preset == "sharegpt":
        return _pair_from_sharegpt(obj)
    if preset == "openai":
        return _pair_from_messages(obj.get("messages"))
    if preset == "alpaca":
        return _pair_from_alpaca(obj)
    if preset == "prompt_response":
        p = obj.get("prompt") or obj.get("question") or obj.get("input")
        r = obj.get("response") or obj.get("answer") or obj.get("output")
        return _clean(p), _clean(r)

    # auto: try shapes in order of specificity
    if isinstance(obj.get("conversations") or obj.get("conversation"), list):
        p, r = _pair_from_sharegpt(obj)
        if p and r:
            return p, r
    if isinstance(obj.get("messages"), list):
        p, r = _pair_from_messages(obj.get("messages"))
        if p and r:
            return p, r
    if obj.get("instruction") is not None or obj.get("instruct") is not None:
        p, r = _pair_from_alpaca(obj)
        if p and r:
            return p, r
    p = obj.get("prompt") or obj.get("question") or obj.get("input")
    r = obj.get("response") or obj.get("answer") or obj.get("output")
    return _clean(p), _clean(r)


def objects_to_rows(objs: Iterable[Any], *, preset: str = "auto") -> List[Row]:
    rows: List[Row] = []
    for obj in objs:
        if not isinstance(obj, dict):
            continue
        p, r = pair_from_obj(obj, preset=preset)
        rows.append({"prompt": p, "response": r})
    return rows_to_training_data(rows)


def ingest_csv(
    path: Path,
    *,
    prompt_col: str = "prompt",
    response_col: str = "response",
    delimiter: Optional[str] = None,
) -> List[Row]:
    delim = delimiter
    if delim is None:
        delim = "\t" if path.suffix.lower() == ".tsv" else ","
    rows: List[Row] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        if not reader.fieldnames:
            return []
        # Case-insensitive column match.
        fields = {n.lower(): n for n in reader.fieldnames if n}
        pc = fields.get(prompt_col.lower())
        rc = fields.get(response_col.lower())
        if not pc or not rc:
            # Common aliases.
            for alt_p in ("prompt", "question", "input", "instruction", "user"):
                if alt_p in fields:
                    pc = pc or fields[alt_p]
            for alt_r in ("response", "answer", "output", "completion", "assistant"):
                if alt_r in fields:
                    rc = rc or fields[alt_r]
        if not pc or not rc:
            raise ValueError(
                f"CSV needs prompt/response columns "
                f"(have {list(reader.fieldnames)}; "
                f"set --prompt-col / --response-col)"
            )
        for raw in reader:
            rows.append(
                {
                    "prompt": _clean(raw.get(pc)),
                    "response": _clean(raw.get(rc)),
                }
            )
    return rows_to_training_data(rows)


def ingest_jsonl(path: Path, *, preset: str = "auto") -> List[Row]:
    objs: List[Any] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            objs.append(obj)
    return objects_to_rows(objs, preset=preset)


def ingest_json(path: Path, *, preset: str = "auto") -> List[Row]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "training_data" in data:
        data = data["training_data"]
    # Single ShareGPT / chat object (not wrapped in a list).
    if isinstance(data, dict):
        return objects_to_rows([data], preset=preset)
    if not isinstance(data, list):
        raise ValueError(
            f"{path}: expected JSON array, object, or {{training_data: […]}}"
        )
    for i, obj in enumerate(data):
        if not isinstance(obj, dict):
            raise ValueError(f"{path}: index {i}: expected object")
    return objects_to_rows(data, preset=preset)


def ingest_text(
    text: str,
    *,
    fmt: str = "json",
    preset: str = "auto",
    filename: str = "upload.json",
) -> List[Row]:
    """In-memory ingest for HTTP upload helper (T11c).

    WHY: POST /api/train/recipes should not require a local path on the server host.
    """
    kind = (fmt or "auto").lower().strip()
    if kind == "auto":
        suf = Path(filename).suffix.lower()
        if suf in (".jsonl", ".ndjson"):
            kind = "jsonl"
        elif suf in (".csv", ".tsv"):
            kind = "csv" if suf == ".csv" else "tsv"
        elif suf in (".txt", ".md"):
            kind = "txt"
        else:
            kind = "json"
    if kind == "jsonl":
        objs: List[Any] = []
        for line_no, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"jsonl:{line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"jsonl:{line_no}: expected object")
            objs.append(obj)
        return objects_to_rows(objs, preset=preset)
    if kind == "json":
        data = json.loads(text) if text.strip() else []
        if isinstance(data, dict) and "training_data" in data:
            data = data["training_data"]
        if isinstance(data, dict):
            return objects_to_rows([data], preset=preset)
        if not isinstance(data, list):
            raise ValueError("expected JSON array, object, or {training_data:[…]}")
        return objects_to_rows(data, preset=preset)
    if kind in ("csv", "tsv"):
        import io

        delim = "\t" if kind == "tsv" else ","
        reader = csv.DictReader(io.StringIO(text), delimiter=delim)
        rows: List[Row] = []
        if not reader.fieldnames:
            return []
        fields = {n.lower(): n for n in reader.fieldnames if n}
        pc = fields.get("prompt") or fields.get("question") or fields.get("input")
        rc = fields.get("response") or fields.get("answer") or fields.get("output")
        if not pc or not rc:
            raise ValueError("CSV needs prompt/response (or alias) columns")
        for raw in reader:
            rows.append(
                {"prompt": _clean(raw.get(pc)), "response": _clean(raw.get(rc))}
            )
        return rows_to_training_data(rows)
    if kind == "txt":
        # Reuse block logic via temp — keep one code path for separators.
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", encoding="utf-8", delete=False
        ) as tf:
            tf.write(text)
            tmp = Path(tf.name)
        try:
            return ingest_txt_pairs(tmp)
        finally:
            tmp.unlink(missing_ok=True)
    raise ValueError(
        f"ingest_text unsupported format {kind!r} "
        "(want json|jsonl|csv|tsv|txt; PDF/DOCX need a file path)"
    )


def ingest_txt_pairs(path: Path, *, separator: str = "\n---\n") -> List[Row]:
    """Blocks separated by blank line; within block, first line prompt, rest response.

    Or use ``separator`` (default ``\\n---\\n``) between prompt/response inside a block.
    """
    text = path.read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    rows: List[Row] = []
    for block in blocks:
        if separator in block:
            left, right = block.split(separator, 1)
            rows.append({"prompt": _clean(left), "response": _clean(right)})
            continue
        lines = block.splitlines()
        if len(lines) < 2:
            continue
        rows.append({"prompt": _clean(lines[0]), "response": _clean("\n".join(lines[1:]))})
    return rows_to_training_data(rows)


def ingest_pdf(path: Path, *, chunk_chars: int = 1200) -> List[Row]:
    """Optional PDF → summarize-style pairs.

    WHY self-distill default: a PDF has no gold assistant text. prompt="Summarize…" /
    response=chunk is honest; operators replace rows when they have real targets.
    WHY clear ImportError: missing pypdf must not look like a corrupt file.
    """
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as e:
        raise ImportError(
            "PDF ingest needs pypdf: pip install pypdf "
            "(same interpreter / .venv-training)"
        ) from e
    reader = PdfReader(str(path))
    rows: List[Row] = []
    for i, page in enumerate(reader.pages):
        raw = _clean(page.extract_text())
        if not raw:
            continue
        chunks = _chunk_text(raw, chunk_chars)
        for j, chunk in enumerate(chunks):
            rows.append(
                {
                    "prompt": f"Summarize the following document excerpt (page {i + 1}, part {j + 1}):",
                    "response": chunk,
                }
            )
    return rows_to_training_data(rows)


def ingest_docx(path: Path, *, chunk_chars: int = 1200) -> List[Row]:
    try:
        import docx  # type: ignore  # python-docx
    except ImportError as e:
        raise ImportError(
            "DOCX ingest needs python-docx: pip install python-docx "
            "(same interpreter / .venv-training)"
        ) from e
    doc = docx.Document(str(path))
    paras = [_clean(p.text) for p in doc.paragraphs if _clean(p.text)]
    raw = "\n\n".join(paras)
    rows: List[Row] = []
    for j, chunk in enumerate(_chunk_text(raw, chunk_chars)):
        rows.append(
            {
                "prompt": f"Summarize the following document excerpt (part {j + 1}):",
                "response": chunk,
            }
        )
    return rows_to_training_data(rows)


def _chunk_text(text: str, chunk_chars: int) -> List[str]:
    if chunk_chars <= 0 or len(text) <= chunk_chars:
        return [text] if text else []
    out: List[str] = []
    start = 0
    while start < len(text):
        out.append(text[start : start + chunk_chars].strip())
        start += chunk_chars
    return [c for c in out if c]


def ingest_file(
    path: Path | str,
    *,
    fmt: Optional[str] = None,
    preset: str = "auto",
    prompt_col: str = "prompt",
    response_col: str = "response",
    chunk_chars: int = 1200,
) -> List[Row]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    kind = detect_format(path, fmt)
    if kind in ("csv", "tsv"):
        return ingest_csv(
            path,
            prompt_col=prompt_col,
            response_col=response_col,
            delimiter="\t" if kind == "tsv" else ",",
        )
    if kind == "jsonl":
        return ingest_jsonl(path, preset=preset)
    if kind == "json":
        return ingest_json(path, preset=preset)
    if kind == "txt":
        return ingest_txt_pairs(path)
    if kind == "pdf":
        return ingest_pdf(path, chunk_chars=chunk_chars)
    if kind == "docx":
        return ingest_docx(path, chunk_chars=chunk_chars)
    raise ValueError(f"unsupported format {kind!r}")


def recipe_result(
    rows: Sequence[Row],
    *,
    model_name: str = "",
    preset: str = "auto",
) -> Dict[str, Any]:
    """HTTP/CLI-friendly envelope (does not submit a train job)."""
    body: Dict[str, Any] = {
        "n_rows": len(rows),
        "training_data": list(rows),
        "preset": normalize_preset(preset),
    }
    if model_name:
        body["model_name"] = model_name
    return body


def write_jsonl(rows: Sequence[Row], path: Path | str) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def write_train_payload(
    rows: Sequence[Row],
    path: Path | str,
    *,
    model_name: str = "",
) -> None:
    """Write a JSON object ready to merge into ``POST /api/train/jobs`` payload."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body: Dict[str, Any] = {"training_data": list(rows)}
    if model_name:
        body["model_name"] = model_name
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main_stdin() -> int:
    """T11c HTTP helper: stdin JSON → stdout recipe_result (no train submit)."""
    try:
        req = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"error: invalid JSON stdin: {e}", file=sys.stderr)
        return 1
    if not isinstance(req, dict):
        print("error: stdin must be a JSON object", file=sys.stderr)
        return 1
    content = req.get("content")
    if content is None:
        print("error: content required", file=sys.stderr)
        return 1
    if not isinstance(content, str):
        content = str(content)
    try:
        rows = ingest_text(
            content,
            fmt=str(req.get("format") or "auto"),
            preset=str(req.get("preset") or "auto"),
            filename=str(req.get("filename") or "upload.json"),
        )
        out = recipe_result(
            rows,
            model_name=str(req.get("model_name") or ""),
            preset=str(req.get("preset") or "auto"),
        )
    except (ValueError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--stdin":
        return main_stdin()

    p = argparse.ArgumentParser(
        description="T11: ingest CSV/JSONL/TXT/(PDF|DOCX) → training_data JSON/JSONL"
    )
    p.add_argument("input", type=Path, help="Source file")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output .jsonl or .json (payload wrapper if .json)",
    )
    p.add_argument("--format", default="auto", help="auto|csv|tsv|jsonl|json|txt|pdf|docx")
    p.add_argument(
        "--preset",
        default="auto",
        help="JSON/JSONL chat-export: auto|sharegpt|openai|alpaca|prompt_response",
    )
    p.add_argument("--prompt-col", default="prompt")
    p.add_argument("--response-col", default="response")
    p.add_argument("--chunk-chars", type=int, default=1200)
    p.add_argument("--model-name", default="", help="Embed in .json payload wrapper")
    args = p.parse_args(argv)

    try:
        rows = ingest_file(
            args.input,
            fmt=args.format,
            preset=args.preset,
            prompt_col=args.prompt_col,
            response_col=args.response_col,
            chunk_chars=args.chunk_chars,
        )
    except (ValueError, FileNotFoundError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    out = args.output
    if out.suffix.lower() == ".json":
        write_train_payload(rows, out, model_name=args.model_name)
    else:
        write_jsonl(rows, out)
    print(f"wrote {len(rows)} rows → {out} (preset={normalize_preset(args.preset)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
