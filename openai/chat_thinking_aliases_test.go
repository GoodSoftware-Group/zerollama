package openai

import (
	"strings"
	"testing"
)

func TestBindChatCompletionRequest_ChatTemplateKwargsEnableThinking(t *testing.T) {
	raw := []byte(`{
		"model": "qwen2.5:0.5b",
		"messages": [{"role":"user","content":"hi"}],
		"chat_template_kwargs": {"enable_thinking": false}
	}`)
	req, err := BindChatCompletionRequest(raw)
	if err != nil {
		t.Fatal(err)
	}
	out, err := FromChatRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if out.Think == nil || out.Think.Bool() {
		t.Fatalf("Think=%v, want false from chat_template_kwargs.enable_thinking", out.Think)
	}
}

func TestBindChatCompletionRequest_BogusChatTemplateKwargRejected(t *testing.T) {
	raw := []byte(`{
		"model": "qwen2.5:0.5b",
		"messages": [{"role":"user","content":"hi"}],
		"chat_template_kwargs": {"bogus_kwarg_zzq": true}
	}`)
	_, err := BindChatCompletionRequest(raw)
	if err == nil {
		t.Fatal("expected unknown nested kwarg error")
	}
	if got := err.Error(); !strings.Contains(got, "chat_template_kwargs") || !strings.Contains(got, "bogus_kwarg_zzq") {
		t.Fatalf("error = %v", err)
	}
}

func TestBindChatCompletionRequest_TopLevelEnableThinking(t *testing.T) {
	raw := []byte(`{
		"model": "qwen2.5:0.5b",
		"messages": [{"role":"user","content":"hi"}],
		"enable_thinking": true
	}`)
	req, err := BindChatCompletionRequest(raw)
	if err != nil {
		t.Fatal(err)
	}
	out, err := FromChatRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if out.Think == nil || !out.Think.Bool() {
		t.Fatalf("Think=%v, want true", out.Think)
	}
}

func TestFromChatRequest_ReasoningEffortHighSetsThinkFromAlias(t *testing.T) {
	effort := "high"
	req := ChatCompletionRequest{
		Model:           "qwen2.5:0.5b",
		Messages:        []Message{{Role: "user", Content: "hi"}},
		ReasoningEffort: &effort,
	}
	out, err := FromChatRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if out.Think == nil || !out.Think.Bool() {
		t.Fatalf("Think=%v", out.Think)
	}
	if !out.ThinkFromAlias {
		t.Fatal("expected ThinkFromAlias so non-thinking models soft-ignore reasoning_effort")
	}
}
