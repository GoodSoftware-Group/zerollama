package server

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestApplyThinkingGate_Deny(t *testing.T) {
	t.Setenv("ZEROLLAMA_THINKING_GATE", "deny")
	think := &api.ThinkValue{Value: true}
	err := applyThinkingGate(&think)
	if err == nil {
		t.Fatal("expected deny error")
	}
	if !think.Bool() {
		t.Fatal("deny should not mutate before abort")
	}
}

func TestApplyThinkingGate_Strip(t *testing.T) {
	t.Setenv("ZEROLLAMA_THINKING_GATE", "strip")
	think := &api.ThinkValue{Value: "high"}
	if err := applyThinkingGate(&think); err != nil {
		t.Fatal(err)
	}
	if think == nil || think.Bool() {
		t.Fatalf("Think=%v, want false", think)
	}
}

func TestApplyThinkingGate_DefaultAllows(t *testing.T) {
	t.Setenv("ZEROLLAMA_THINKING_GATE", "")
	think := &api.ThinkValue{Value: true}
	if err := applyThinkingGate(&think); err != nil {
		t.Fatal(err)
	}
	if !think.Bool() {
		t.Fatal("default must allow thinking")
	}
}
