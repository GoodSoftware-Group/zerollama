"""Parse Ollama ``format`` / OpenAI ``response_format`` into llama-server constraints.

WHY: Go runtime proxies used to drop format → unconstrained decode on Python path
(M15f). llama-server accepts ``json_schema`` or ``grammar`` on /completion.
"""

from __future__ import annotations

import json
from typing import Any


def parse_format_constraint(
    format_value: Any = None,
    *,
    response_format: Any = None,
) -> dict[str, Any]:
    """Return ``{"json_schema": ..., "grammar": ...}`` keys for /completion.

    Accepts:
    - ``"json"`` → builtin JSON object grammar (caller may map to grammar string)
    - raw JSON Schema object / dict
    - ``{"type":"gbnf","grammar":"..."}``
    - OpenAI ``response_format``: ``json_object`` / ``json_schema``
    """
    out: dict[str, Any] = {}
    if response_format is not None and format_value is None:
        format_value = _from_openai_response_format(response_format)
    if format_value is None:
        return out
    if isinstance(format_value, str):
        raw = format_value.strip()
        if not raw or raw == "null":
            return out
        if raw == "json" or raw == '"json"':
            out["want_json"] = True
            return out
        try:
            format_value = json.loads(raw)
        except json.JSONDecodeError:
            return out
    if isinstance(format_value, dict):
        typ = str(format_value.get("type") or "").strip().lower()
        if typ == "gbnf":
            grammar = format_value.get("grammar")
            if isinstance(grammar, str) and grammar.strip():
                out["grammar"] = grammar
            return out
        # JSON Schema object (including {"type":"object",...}).
        out["json_schema"] = format_value
        return out
    return out


def _from_openai_response_format(rf: Any) -> Any:
    if not isinstance(rf, dict):
        return None
    typ = str(rf.get("type") or "").strip().lower()
    if typ == "json_object":
        return "json"
    if typ == "json_schema":
        js = rf.get("json_schema")
        if isinstance(js, dict) and "schema" in js:
            return js.get("schema")
        return js
    return None


def merge_format_into_options(
    options: dict[str, Any] | None,
    *,
    format_value: Any = None,
    response_format: Any = None,
) -> dict[str, Any]:
    """Copy options and attach ``json_schema`` / ``grammar`` / ``want_json``."""
    opts = dict(options or {})
    parsed = parse_format_constraint(format_value, response_format=response_format)
    for k, v in parsed.items():
        if k not in opts:
            opts[k] = v
    return opts


def apply_format_to_completion_payload(
    payload: dict[str, Any],
    options: dict[str, Any] | None,
) -> None:
    """Mutate llama-server /completion payload with grammar constraints from options."""
    if not options:
        return
    if grammar := options.get("grammar"):
        if isinstance(grammar, str) and grammar.strip():
            payload["grammar"] = grammar
            return
    if schema := options.get("json_schema"):
        payload["json_schema"] = schema
        return
    if options.get("want_json"):
        # Builtin JSON object grammar — same intent as Go llm.grammarJSON.
        payload["grammar"] = (
            "root   ::= object\n"
            'value  ::= object | array | string | number | ("true" | "false" | "null") ws\n'
            "object ::=\n"
            '  "{" ws (\n'
            '         string ":" ws value\n'
            '    ("," ws string ":" ws value)*\n'
            '  )? ws "}"\n'
            "array  ::=\n"
            '  "[" ws (\n'
            "            value\n"
            '    ("," ws value)*\n'
            '  )? ws "]"\n'
            "string ::=\n"
            '  "\\"" (\n'
            '    [^"\\\\\\x7F\\x00-\\x1F] |\n'
            '    "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F]{4})\n'
            '  )* "\\""\n'
            'number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?\n'
            "ws ::= ([ \\t\\n]*)\n"
        )
