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


def ingest_jsonl(path: Path) -> List[Row]:
    rows: List[Row] = []
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
            p = obj.get("prompt") or obj.get("question") or obj.get("input")
            r = obj.get("response") or obj.get("answer") or obj.get("output")
            # Chat messages shape: [{role, content}, …]
            if (not p or not r) and isinstance(obj.get("messages"), list):
                user, asst = "", ""
                for m in obj["messages"]:
                    if not isinstance(m, dict):
                        continue
                    role = _clean(m.get("role")).lower()
                    content = _clean(m.get("content"))
                    if role == "user" and content:
                        user = content
                    elif role in ("assistant", "model") and content:
                        asst = content
                p, r = user or p, asst or r
            rows.append({"prompt": _clean(p), "response": _clean(r)})
    return rows_to_training_data(rows)


def ingest_json(path: Path) -> List[Row]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "training_data" in data:
        data = data["training_data"]
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected JSON array or {{training_data: […]}}")
    # Reuse jsonl object rules via a temp pass.
    rows: List[Row] = []
    for i, obj in enumerate(data):
        if not isinstance(obj, dict):
            raise ValueError(f"{path}: index {i}: expected object")
        p = obj.get("prompt") or obj.get("question") or obj.get("input")
        r = obj.get("response") or obj.get("answer") or obj.get("output")
        if (not p or not r) and isinstance(obj.get("messages"), list):
            user, asst = "", ""
            for m in obj["messages"]:
                if not isinstance(m, dict):
                    continue
                role = _clean(m.get("role")).lower()
                content = _clean(m.get("content"))
                if role == "user" and content:
                    user = content
                elif role in ("assistant", "model") and content:
                    asst = content
            p, r = user or p, asst or r
        rows.append({"prompt": _clean(p), "response": _clean(r)})
    return rows_to_training_data(rows)


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
        return ingest_jsonl(path)
    if kind == "json":
        return ingest_json(path)
    if kind == "txt":
        return ingest_txt_pairs(path)
    if kind == "pdf":
        return ingest_pdf(path, chunk_chars=chunk_chars)
    if kind == "docx":
        return ingest_docx(path, chunk_chars=chunk_chars)
    raise ValueError(f"unsupported format {kind!r}")


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


def main(argv: Optional[Sequence[str]] = None) -> int:
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
    p.add_argument("--prompt-col", default="prompt")
    p.add_argument("--response-col", default="response")
    p.add_argument("--chunk-chars", type=int, default=1200)
    p.add_argument("--model-name", default="", help="Embed in .json payload wrapper")
    args = p.parse_args(argv)

    try:
        rows = ingest_file(
            args.input,
            fmt=args.format,
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
    print(f"wrote {len(rows)} rows → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
