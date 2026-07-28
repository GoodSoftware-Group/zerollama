package api

import (
	"strings"
	"testing"
)

func TestCheckUnknownChatFields_RejectsMinefieldProbe(t *testing.T) {
	err := CheckUnknownChatFields([]byte(`{
		"model": "qwen2.5:0.5b",
		"messages": [{"role":"user","content":"hi"}],
		"__minefield_unvalidated_field_probe__": true
	}`))
	if err == nil {
		t.Fatal("expected unknown field error")
	}
	if !strings.Contains(err.Error(), "__minefield_unvalidated_field_probe__") {
		t.Fatalf("error = %v", err)
	}
}

func TestCheckUnknownChatFields_AllowsKnown(t *testing.T) {
	err := CheckUnknownChatFields([]byte(`{
		"model": "qwen2.5:0.5b",
		"messages": [{"role":"user","content":"hi"}],
		"stream": false,
		"think": false,
		"options": {"temperature": 0},
		"keep_alive": "5m"
	}`))
	if err != nil {
		t.Fatal(err)
	}
}

func TestCheckUnknownChatFields_AllowsChatTemplateKwargs(t *testing.T) {
	err := CheckUnknownChatFields([]byte(`{
		"model": "qwen2.5:0.5b",
		"messages": [{"role":"user","content":"hi"}],
		"chat_template_kwargs": {"enable_thinking": true}
	}`))
	if err != nil {
		t.Fatal(err)
	}
}

func TestApplyChatThinkingAliases_EnableThinking(t *testing.T) {
	on := false
	req := &ChatRequest{EnableThinking: &on}
	if err := ApplyChatThinkingAliases(req); err != nil {
		t.Fatal(err)
	}
	if req.Think == nil || req.Think.Bool() {
		t.Fatalf("Think=%v", req.Think)
	}
}

func TestApplyChatThinkingAliases_BogusKwarg(t *testing.T) {
	req := &ChatRequest{ChatTemplateKwargs: map[string]any{"bogus_kwarg_zzq": true}}
	err := ApplyChatThinkingAliases(req)
	if err == nil || !strings.Contains(err.Error(), "bogus_kwarg_zzq") {
		t.Fatalf("err=%v", err)
	}
}

func TestCheckUnknownGenerateFields_RejectsMinefieldProbe(t *testing.T) {
	err := CheckUnknownGenerateFields([]byte(`{
		"model": "qwen2.5:0.5b",
		"prompt": "hi",
		"__minefield_unvalidated_field_probe__": true
	}`))
	if err == nil {
		t.Fatal("expected unknown field error")
	}
	if !strings.Contains(err.Error(), "__minefield_unvalidated_field_probe__") {
		t.Fatalf("error = %v", err)
	}
}

func TestCheckUnknownGenerateFields_AllowsKnown(t *testing.T) {
	err := CheckUnknownGenerateFields([]byte(`{
		"model": "qwen2.5:0.5b",
		"prompt": "hi",
		"stream": false,
		"think": false,
		"enable_thinking": false,
		"options": {"temperature": 0}
	}`))
	if err != nil {
		t.Fatal(err)
	}
}

func TestApplyGenerateThinkingAliases(t *testing.T) {
	on := true
	req := &GenerateRequest{EnableThinking: &on}
	if err := ApplyGenerateThinkingAliases(req); err != nil {
		t.Fatal(err)
	}
	if req.Think == nil || !req.Think.Bool() {
		t.Fatalf("Think=%v", req.Think)
	}
}
