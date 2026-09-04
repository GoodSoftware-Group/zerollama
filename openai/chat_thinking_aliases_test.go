package openai

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
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

func TestFromChatRequest_ReasoningBudgetOutranksEffort(t *testing.T) {
	effort := "high"
	zero := 0
	req := ChatCompletionRequest{
		Model:                 "qwen2.5:0.5b",
		Messages:              []Message{{Role: "user", Content: "hi"}},
		ReasoningEffort:       &effort,
		ReasoningBudgetTokens: &zero,
	}
	out, err := FromChatRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if out.Think == nil || out.Think.Bool() {
		t.Fatalf("budget 0 should disable think, got %v", out.Think)
	}
	if !out.ThinkFromAlias {
		t.Fatal("budget is an alias")
	}

	on := 128
	off := false
	req = ChatCompletionRequest{
		Model:                 "qwen2.5:0.5b",
		Messages:              []Message{{Role: "user", Content: "hi"}},
		EnableThinking:        &off,
		ReasoningBudgetTokens: &on,
	}
	out, err = FromChatRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if out.Think == nil || !out.Think.Bool() {
		t.Fatalf("budget >0 should enable think over enable_thinking=false, got %v", out.Think)
	}
}

func TestFromChatRequest_ThinkWinsOverBudget(t *testing.T) {
	on := 128
	req := ChatCompletionRequest{
		Model:                 "qwen2.5:0.5b",
		Messages:              []Message{{Role: "user", Content: "hi"}},
		Think:                 &api.ThinkValue{Value: false},
		ReasoningBudgetTokens: &on,
	}
	out, err := FromChatRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if out.Think == nil || out.Think.Bool() {
		t.Fatalf("native think=false must win, got %v", out.Think)
	}
	if out.ThinkFromAlias {
		t.Fatal("native think is not an alias")
	}
}

func TestBindChatCompletionRequest_ReasoningBudgetExtraBody(t *testing.T) {
	raw := []byte(`{
		"model": "qwen2.5:0.5b",
		"messages": [{"role":"user","content":"hi"}],
		"reasoning_effort": "high",
		"extra_body": {"reasoning_budget_tokens": 0}
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
		t.Fatalf("extra_body budget 0 should disable, got %v", out.Think)
	}
}

func TestFromChatRequest_ReasoningBudgetNegative(t *testing.T) {
	neg := -1
	_, err := FromChatRequest(ChatCompletionRequest{
		Model:                 "m",
		Messages:              []Message{{Role: "user", Content: "hi"}},
		ReasoningBudgetTokens: &neg,
	})
	if err == nil || !strings.Contains(err.Error(), "reasoning_budget_tokens") {
		t.Fatalf("err=%v", err)
	}
}
