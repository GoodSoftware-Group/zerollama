package server

import (
	"encoding/json"
	"testing"

	"github.com/ollama/ollama/openai"
)

func TestV1ThinkNeedsLegacy(t *testing.T) {
	t.Parallel()
	if v1ThinkNeedsLegacy(false) || v1ThinkNeedsLegacy(nil) {
		t.Fatal("think false/nil should not require legacy")
	}
	if !v1ThinkNeedsLegacy(true) {
		t.Fatal("think true should require legacy")
	}
	if !v1ThinkNeedsLegacy("high") {
		t.Fatal("think high should require legacy")
	}
	if v1ThinkNeedsLegacy("  ") {
		t.Fatal("think empty string should not require legacy")
	}
}

func TestV1BodyNeedsLegacyRunner(t *testing.T) {
	t.Parallel()
	if v1BodyNeedsLegacyRunner(map[string]any{"think": false}) {
		t.Fatal("think false should not require legacy")
	}
	if !v1BodyNeedsLegacyRunner(map[string]any{"think": true}) {
		t.Fatal("think true should require legacy")
	}
	if !v1BodyNeedsLegacyRunner(map[string]any{"reasoning_effort": "high"}) {
		t.Fatal("reasoning_effort should require legacy")
	}
	if !v1BodyNeedsLegacyRunner(map[string]any{"reasoning": map[string]any{"effort": "low"}}) {
		t.Fatal("reasoning object should require legacy")
	}
	if v1BodyNeedsLegacyRunner(map[string]any{"messages": []any{}}) {
		t.Fatal("plain body should not require legacy")
	}
}

func TestV1ChatNeedsLegacyRunnerReasoning(t *testing.T) {
	t.Parallel()
	trueVal := true
	effort := "medium"
	req := &openai.ChatCompletionRequest{
		Messages:  []openai.Message{{Role: "user", Content: "hi"}},
		Reasoning: &openai.Reasoning{Effort: "low"},
	}
	if !v1ChatNeedsLegacyRunner(req, nil) {
		t.Fatal("Reasoning block should require legacy")
	}
	req = &openai.ChatCompletionRequest{
		Messages:        []openai.Message{{Role: "user", Content: "hi"}},
		ReasoningEffort: &effort,
	}
	if !v1ChatNeedsLegacyRunner(req, nil) {
		t.Fatal("reasoning_effort should require legacy")
	}
	req = &openai.ChatCompletionRequest{
		Messages: []openai.Message{{
			Role:      "assistant",
			Content:   "",
			Reasoning: "chain of thought",
		}},
	}
	if !v1ChatNeedsLegacyRunner(req, nil) {
		t.Fatal("message reasoning should require legacy")
	}
	req = &openai.ChatCompletionRequest{
		Messages: []openai.Message{{Role: "user", Content: "hi"}},
		Logprobs: &trueVal,
	}
	if !v1ChatNeedsLegacyRunner(req, nil) {
		t.Fatal("logprobs should require legacy")
	}
	raw := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f"}}]}`)
	var withTools openai.ChatCompletionRequest
	if err := json.Unmarshal(raw, &withTools); err != nil {
		t.Fatal(err)
	}
	var bodyMap map[string]any
	if err := json.Unmarshal(raw, &bodyMap); err != nil {
		t.Fatal(err)
	}
	if v1ChatNeedsLegacyRunner(&withTools, bodyMap) {
		t.Fatal("plain text + tools should use runtime")
	}
}
