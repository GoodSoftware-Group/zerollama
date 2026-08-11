"""Tests for M15f format → llama-server grammar/json_schema forwarding."""

from __future__ import annotations

from runtime.format_constraint import (
    apply_format_to_completion_payload,
    merge_format_into_options,
    parse_format_constraint,
)


def test_parse_json_schema():
    out = parse_format_constraint({"type": "object", "properties": {"a": {"type": "string"}}})
    assert "json_schema" in out
    assert out["json_schema"]["type"] == "object"


def test_parse_gbnf():
    out = parse_format_constraint({"type": "gbnf", "grammar": 'root ::= "hi"'})
    assert out.get("grammar") == 'root ::= "hi"'


def test_parse_openai_response_format():
    out = parse_format_constraint(
        response_format={
            "type": "json_schema",
            "json_schema": {"schema": {"type": "object"}},
        }
    )
    assert out["json_schema"]["type"] == "object"


def test_apply_format_to_payload():
    payload: dict = {"prompt": "x"}
    opts = merge_format_into_options({}, format_value={"type": "gbnf", "grammar": "root ::= \"a\""})
    apply_format_to_completion_payload(payload, opts)
    assert payload["grammar"] == 'root ::= "a"'

    payload2: dict = {"prompt": "x"}
    opts2 = merge_format_into_options(
        {},
        response_format={"type": "json_schema", "json_schema": {"schema": {"type": "object"}}},
    )
    apply_format_to_completion_payload(payload2, opts2)
    assert payload2["json_schema"]["type"] == "object"
