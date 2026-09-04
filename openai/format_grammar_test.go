package openai

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestFromChatRequest_RejectsToolsWithGrammar(t *testing.T) {
	tools := json.RawMessage(`[{"type":"function","function":{"name":"x","parameters":{"type":"object"}}}]`)
	var toolList api.Tools
	if err := json.Unmarshal(tools, &toolList); err != nil {
		t.Fatal(err)
	}
	_, err := FromChatRequest(ChatCompletionRequest{
		Model:    "m",
		Messages: []Message{{Role: "user", Content: "hi"}},
		Tools:    toolList,
		Format:   json.RawMessage(`{"type":"gbnf","grammar":"root ::= \"a\""}`),
	})
	if err == nil || !strings.Contains(err.Error(), "grammar is not supported together with tools") {
		t.Fatalf("err=%v", err)
	}
}

func TestFromChatRequest_FormatGBNF(t *testing.T) {
	req, err := FromChatRequest(ChatCompletionRequest{
		Model:    "m",
		Messages: []Message{{Role: "user", Content: "hi"}},
		Format:   json.RawMessage(`{"type":"gbnf","grammar":"root ::= \"a\""}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	if string(req.Format) != `{"type":"gbnf","grammar":"root ::= \"a\""}` {
		t.Fatalf("format=%s", req.Format)
	}
}

func TestFormatHasGrammar(t *testing.T) {
	if !formatHasGrammar(json.RawMessage(`"json"`)) {
		t.Fatal("json")
	}
	if !formatHasGrammar(json.RawMessage(`{"type":"object"}`)) {
		t.Fatal("schema")
	}
	if formatHasGrammar(nil) {
		t.Fatal("nil")
	}
}

func TestFromChatRequest_JsonObjectAndFlatSchema(t *testing.T) {
	obj, err := FromChatRequest(ChatCompletionRequest{
		Model:          "m",
		Messages:       []Message{{Role: "user", Content: "hi"}},
		ResponseFormat: &ResponseFormat{Type: "json_object"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if string(obj.Format) != `"json"` {
		t.Fatalf("json_object format=%s", obj.Format)
	}
	flat, err := FromChatRequest(ChatCompletionRequest{
		Model:    "m",
		Messages: []Message{{Role: "user", Content: "hi"}},
		ResponseFormat: &ResponseFormat{
			Type:   "json_schema",
			Schema: json.RawMessage(`{"type":"object"}`),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if string(flat.Format) != `{"type":"object"}` {
		t.Fatalf("flat schema format=%s", flat.Format)
	}
}
