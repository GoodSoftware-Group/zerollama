package openai

import (
	"encoding/json"
	"strings"
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

// TestBindChatCompletionRequest_UnknownTopLevelFieldRejected locks minefield trap 77:
// invented top-level fields must 400 so typos / wrong knobs are loud, not silent.
func TestBindChatCompletionRequest_UnknownTopLevelFieldRejected(t *testing.T) {
	raw := []byte(`{
		"model": "qwen2.5:0.5b",
		"messages": [{"role": "user", "content": "hi"}],
		"__minefield_unvalidated_field_probe__": true
	}`)
	_, err := BindChatCompletionRequest(raw)
	if err == nil {
		t.Fatal("expected unknown field error")
	}
	if !strings.Contains(err.Error(), "__minefield_unvalidated_field_probe__") {
		t.Fatalf("error = %v, want probe field name", err)
	}
}

func TestBindChatCompletionRequest_KnownFieldsStillOK(t *testing.T) {
	raw := []byte(`{
		"model": "qwen2.5:0.5b",
		"messages": [{"role": "user", "content": "hi"}],
		"temperature": 0,
		"user": "minefield",
		"extra_body": {"prompt_cache_key": "k"}
	}`)
	req, err := BindChatCompletionRequest(raw)
	if err != nil {
		t.Fatal(err)
	}
	if req.Model != "qwen2.5:0.5b" {
		t.Fatalf("model = %q", req.Model)
	}
	if req.PromptCacheKey == nil || *req.PromptCacheKey != "k" {
		t.Fatalf("PromptCacheKey=%v", req.PromptCacheKey)
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
