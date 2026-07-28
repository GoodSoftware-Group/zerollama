package openai

import (
	"encoding/json"
	"testing"
)

// TestChatCompletionRequest_ToolChoiceNoneOmitsTools locks minefield trap 78 fix:
// tool_choice "none" must strip tools from the underlying chat request.
func TestChatCompletionRequest_ToolChoiceNoneOmitsTools(t *testing.T) {
	raw := []byte(`{
		"model": "qwen2.5:0.5b",
		"messages": [{"role": "user", "content": "Call get_time"}],
		"tools": [{
			"type": "function",
			"function": {
				"name": "get_time",
				"description": "time",
				"parameters": {"type": "object", "properties": {}}
			}
		}],
		"tool_choice": "none"
	}`)

	var req ChatCompletionRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatal(err)
	}
	if len(req.Tools) != 1 {
		t.Fatalf("tools len=%d after unmarshal, want 1", len(req.Tools))
	}
	if !toolChoiceMeansNone(req.ToolChoice) {
		t.Fatalf("ToolChoice=%v want none", req.ToolChoice)
	}

	out, err := FromChatRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Tools) != 0 {
		t.Fatalf("FromChatRequest tools len=%d, want 0 when tool_choice=none (trap 78)", len(out.Tools))
	}
}

func TestChatCompletionRequest_ToolChoiceAutoKeepsTools(t *testing.T) {
	raw := []byte(`{
		"model": "qwen2.5:0.5b",
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [{
			"type": "function",
			"function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}
		}],
		"tool_choice": "auto"
	}`)
	var req ChatCompletionRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatal(err)
	}
	out, err := FromChatRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Tools) != 1 {
		t.Fatalf("tools len=%d, want 1 for tool_choice=auto", len(out.Tools))
	}
}

func TestFromResponsesRequest_ToolChoiceNoneOmitsTools(t *testing.T) {
	desc := "time"
	req := ResponsesRequest{
		Model: "qwen2.5:0.5b",
		Input: ResponsesInput{Text: "Call get_time"},
		Tools: []ResponsesTool{{
			Type:        "function",
			Name:        "get_time",
			Description: &desc,
			Parameters:  map[string]any{"type": "object", "properties": map[string]any{}},
		}},
		ToolChoice: "none",
	}
	out, err := FromResponsesRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Tools) != 0 {
		t.Fatalf("tools len=%d, want 0", len(out.Tools))
	}
}

// TestChatCompletionRequest_UnknownTopLevelFieldAccepted documents minefield trap 77:
// invented top-level fields do not fail unmarshal (OpenAI-compat). HTTP 200 alone
// cannot confirm the request surface was validated — assert on response fields.
func TestChatCompletionRequest_UnknownTopLevelFieldAccepted(t *testing.T) {
	raw := []byte(`{
		"model": "qwen2.5:0.5b",
		"messages": [{"role": "user", "content": "hi"}],
		"__minefield_unvalidated_field_probe__": true
	}`)
	var req ChatCompletionRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatalf("unknown field should be ignored, not error: %v", err)
	}
	out, err := FromChatRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if out.Model != "qwen2.5:0.5b" {
		t.Fatalf("converted model = %q", out.Model)
	}
}

func TestToolChoiceMeansNone(t *testing.T) {
	if !toolChoiceMeansNone("none") || !toolChoiceMeansNone("NONE") {
		t.Fatal("string none")
	}
	if toolChoiceMeansNone("auto") || toolChoiceMeansNone("required") {
		t.Fatal("auto/required must not mean none")
	}
	if toolChoiceMeansNone(map[string]any{"type": "function", "name": "x"}) {
		t.Fatal("object form is not none")
	}
	if !toolChoiceMeansNone(json.RawMessage(`"none"`)) {
		t.Fatal("RawMessage none")
	}
}
