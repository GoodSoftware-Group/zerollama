package template

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestApplyContinueFinalChatML(t *testing.T) {
	msgs := []api.Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "Hello world  "},
	}
	rendered := "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nHello world  <|im_end|>\n<|im_start|>assistant\n"
	got, ok := ApplyContinueFinal(rendered, msgs)
	if !ok {
		t.Fatal("expected continue")
	}
	want := "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nHello world"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestApplyContinueFinalRejectsTools(t *testing.T) {
	msgs := []api.Message{
		{Role: "assistant", Content: "x", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "fn"}}}},
	}
	if _, ok := ApplyContinueFinal("x", msgs); ok {
		t.Fatal("tool-call replies are not continuable")
	}
}

func TestContinueFinalAllowedUserLast(t *testing.T) {
	if ContinueFinalAllowed([]api.Message{{Role: "user", Content: "hi"}}) {
		t.Fatal("user last is a new turn")
	}
}
