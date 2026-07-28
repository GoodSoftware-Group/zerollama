package envconfig

import "testing"

func TestThinkingGate(t *testing.T) {
	t.Setenv("ZEROLLAMA_THINKING_GATE", "")
	if ThinkingGate() != "" {
		t.Fatalf("empty → %q", ThinkingGate())
	}
	t.Setenv("ZEROLLAMA_THINKING_GATE", "deny")
	if ThinkingGate() != "deny" {
		t.Fatalf("deny → %q", ThinkingGate())
	}
	t.Setenv("ZEROLLAMA_THINKING_GATE", "STRIP")
	if ThinkingGate() != "strip" {
		t.Fatalf("STRIP → %q", ThinkingGate())
	}
	t.Setenv("ZEROLLAMA_THINKING_GATE", "bogus")
	if ThinkingGate() != "" {
		t.Fatalf("bogus → %q", ThinkingGate())
	}
}
