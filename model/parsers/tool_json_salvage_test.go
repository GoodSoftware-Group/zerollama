package parsers

import (
	"fmt"
	"strings"
	"testing"
)

func TestSalvageJSONToolCallTruncatedArgs(t *testing.T) {
	call, ok := salvageJSONToolCall(`{"name": "get_weather", "arguments": {"loc`)
	if !ok {
		t.Fatal("expected salvage")
	}
	if call.Function.Name != "get_weather" {
		t.Fatalf("name=%q", call.Function.Name)
	}
	if call.Function.Arguments.Len() != 0 {
		t.Fatal("truncated args must ship empty object, not partial values")
	}
}

func TestSalvageGLM46ToolCall(t *testing.T) {
	call, ok := salvageGLM46ToolCall("<tool_call>write\n<arg_key>content</arg_key>\n<arg_value>partial")
	if !ok || call.Function.Name != "write" || call.Function.Arguments.Len() != 0 {
		t.Fatalf("got %+v ok=%v", call, ok)
	}
	if _, ok := salvageGLM46ToolCall("<tool_call>"); ok {
		t.Fatal("empty body")
	}
}

func TestSalvageJSONToolCallNoName(t *testing.T) {
	if _, ok := salvageJSONToolCall(`{"arguments": {"x": 1}`); ok {
		t.Fatal("no name")
	}
}

func TestParseJSONToolCallXMLFunctionFirst(t *testing.T) {
	raw := `<function=write_file><parameter=content>
{"name": "evil", "arguments": {"x": 1}}
</parameter></function>`
	got, err := parseJSONToolCall(qwenEventRawToolCall{raw: raw}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got.Function.Name != "write_file" {
		t.Fatalf("name=%q, JSON branch must not steal package.json name", got.Function.Name)
	}
	v, ok := got.Function.Arguments.Get("content")
	if !ok {
		t.Fatal("missing content param")
	}
	if !strings.Contains(fmt.Sprint(v), `"name": "evil"`) {
		t.Fatalf("content=%v", v)
	}
}

func TestParseQwen3ToolCallXMLFunctionFirst(t *testing.T) {
	raw := `<function=write_file><parameter=path>a.txt</parameter></function>`
	got, err := parseQwen3ToolCall(qwen3EventRawToolCall{raw: raw}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got.Function.Name != "write_file" {
		t.Fatalf("name=%q", got.Function.Name)
	}
}

func TestParseQwen3ToolCallSalvagesTruncation(t *testing.T) {
	got, err := parseQwen3ToolCall(qwen3EventRawToolCall{raw: `{"name":"search","arguments":{"q":`}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got.Function.Name != "search" || got.Function.Arguments.Len() != 0 {
		t.Fatalf("%+v", got.Function)
	}
}
