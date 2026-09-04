package openai

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
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
	if strings.Contains(out.Messages[0].Content, api.RequiredToolCallHint) {
		t.Fatalf("auto must not hint, got %q", out.Messages[0].Content)
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

func TestFromResponsesRequest_ToolChoiceNamedFunction(t *testing.T) {
	desc := "time"
	req := ResponsesRequest{
		Model: "m",
		Input: ResponsesInput{Text: "hi"},
		Tools: []ResponsesTool{
			{Type: "function", Name: "get_time", Description: &desc, Parameters: map[string]any{"type": "object"}},
			{Type: "function", Name: "get_weather", Description: &desc, Parameters: map[string]any{"type": "object"}},
		},
		ToolChoice: map[string]any{"type": "function", "name": "get_weather"},
	}
	out, err := FromResponsesRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Tools) != 1 || out.Tools[0].Function.Name != "get_weather" {
		t.Fatalf("tools=%v", out.Tools)
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

func TestFromChatRequest_ServiceTier(t *testing.T) {
	ok := ChatCompletionRequest{Model: "m", Messages: []Message{{Role: "user", Content: "hi"}}, ServiceTier: "auto"}
	if _, err := FromChatRequest(ok); err != nil {
		t.Fatal(err)
	}
	bad := ok
	bad.ServiceTier = "flex"
	_, err := FromChatRequest(bad)
	if err == nil || !strings.Contains(err.Error(), "service_tier") {
		t.Fatalf("err=%v", err)
	}
}

func TestFromChatRequest_LegacyFunctionsMapToTools(t *testing.T) {
	req, err := BindChatCompletionRequest([]byte(`{
		"model": "qwen2.5:0.5b",
		"messages": [{"role": "user", "content": "time?"}],
		"functions": [{
			"name": "get_time",
			"description": "time",
			"parameters": {"type": "object", "properties": {}}
		}],
		"function_call": {"name": "get_time"}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	out, err := FromChatRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Tools) != 1 || out.Tools[0].Function.Name != "get_time" {
		t.Fatalf("tools=%v", out.Tools)
	}
}

func TestFromChatRequest_ToolsWinOverFunctions(t *testing.T) {
	req, err := BindChatCompletionRequest([]byte(`{
		"model": "qwen2.5:0.5b",
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [{
			"type": "function",
			"function": {"name": "keep_me", "parameters": {"type": "object", "properties": {}}}
		}],
		"functions": [{
			"name": "drop_me",
			"parameters": {"type": "object", "properties": {}}
		}]
	}`))
	if err != nil {
		t.Fatal(err)
	}
	out, err := FromChatRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Tools) != 1 || out.Tools[0].Function.Name != "keep_me" {
		t.Fatalf("tools=%v", out.Tools)
	}
}

func TestFromChatRequest_FunctionCallNoneOmitsTools(t *testing.T) {
	req, err := BindChatCompletionRequest([]byte(`{
		"model": "qwen2.5:0.5b",
		"messages": [{"role": "user", "content": "hi"}],
		"functions": [{"name": "get_time", "parameters": {"type": "object", "properties": {}}}],
		"function_call": "none"
	}`))
	if err != nil {
		t.Fatal(err)
	}
	out, err := FromChatRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Tools) != 0 {
		t.Fatalf("tools len=%d, want 0 for function_call=none", len(out.Tools))
	}
}

func TestFromChatRequest_ExtraBodyFunctions(t *testing.T) {
	req, err := BindChatCompletionRequest([]byte(`{
		"model": "qwen2.5:0.5b",
		"messages": [{"role": "user", "content": "hi"}],
		"extra_body": {
			"functions": [{"name": "get_time", "parameters": {"type": "object", "properties": {}}}]
		}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	out, err := FromChatRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Tools) != 1 || out.Tools[0].Function.Name != "get_time" {
		t.Fatalf("tools=%v", out.Tools)
	}
}

func TestFromChatRequest_LegacyFunctionMissingName(t *testing.T) {
	req := ChatCompletionRequest{
		Model:     "m",
		Messages:  []Message{{Role: "user", Content: "hi"}},
		Functions: []api.ToolFunction{{Parameters: api.ToolFunctionParameters{Type: "object"}}},
	}
	_, err := FromChatRequest(req)
	if err == nil || !strings.Contains(err.Error(), "functions[0].name") {
		t.Fatalf("err=%v", err)
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

func TestFromChatRequest_ToolChoiceNamedFunction(t *testing.T) {
	raw := []byte(`{
		"model": "m",
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [
			{"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}},
			{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}}
		],
		"tool_choice": {"type": "function", "function": {"name": "get_weather"}}
	}`)
	var req ChatCompletionRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatal(err)
	}
	out, err := FromChatRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Tools) != 1 || out.Tools[0].Function.Name != "get_weather" {
		t.Fatalf("tools=%v", out.Tools)
	}
	if !strings.Contains(out.Messages[0].Content, api.RequiredToolCallHint) {
		t.Fatalf("named tool_choice should hint, got %q", out.Messages[0].Content)
	}
}

func TestFromChatRequest_ToolChoiceRequiredKeepsTools(t *testing.T) {
	raw := []byte(`{
		"model": "m",
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [
			{"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}}
		],
		"tool_choice": "required"
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
		t.Fatalf("required must keep tools, got %d", len(out.Tools))
	}
	if !strings.Contains(out.Messages[0].Content, api.RequiredToolCallHint) {
		t.Fatalf("required should hint, got %q", out.Messages[0].Content)
	}
}

func TestFromChatRequest_ToolChoiceUnknownName(t *testing.T) {
	req := ChatCompletionRequest{
		Model:    "m",
		Messages: []Message{{Role: "user", Content: "hi"}},
		Tools: []api.Tool{{
			Type:     "function",
			Function: api.ToolFunction{Name: "get_time"},
		}},
		ToolChoice: map[string]any{
			"type":     "function",
			"function": map[string]any{"name": "nope"},
		},
	}
	_, err := FromChatRequest(req)
	if err == nil || !strings.Contains(err.Error(), "unknown function") {
		t.Fatalf("err=%v", err)
	}
}

func TestFromChatRequest_ToolChoiceUnsupportedType(t *testing.T) {
	_, err := FromChatRequest(ChatCompletionRequest{
		Model:      "m",
		Messages:   []Message{{Role: "user", Content: "hi"}},
		ToolChoice: map[string]any{"type": "allowed_tools"},
	})
	if err == nil || !strings.Contains(err.Error(), "unsupported tool_choice.type") {
		t.Fatalf("err=%v", err)
	}
}
