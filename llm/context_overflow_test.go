package llm

import "testing"

func TestContextOverflowMessage(t *testing.T) {
	got := ContextOverflowMessage(9000, 8192)
	if got != "input length (9000 tokens) exceeds the model's maximum context length (8192 tokens)" {
		t.Fatalf("got %q", got)
	}
}
