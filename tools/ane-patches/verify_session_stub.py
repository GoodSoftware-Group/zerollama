#!/usr/bin/env python3
"""Fail if ane_draft_session_stub.cpp does not implement all C API in ane_draft_session.h."""
from __future__ import annotations

import re
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} ane_draft_session.h ane_draft_session_stub.cpp")
    h = Path(sys.argv[1]).read_text()
    stub = Path(sys.argv[2]).read_text()
    decls = re.findall(
        r"^(?:bool|void|int|size_t|uint32_t)\s+(ane_draft_session_\w+)\s*\(",
        h,
        re.MULTILINE,
    )
    missing = [d for d in decls if not re.search(rf"\b{d}\s*\(", stub)]
    if missing:
        raise SystemExit(f"ane_draft_session_stub.cpp missing: {', '.join(missing)}")


if __name__ == "__main__":
    main()
