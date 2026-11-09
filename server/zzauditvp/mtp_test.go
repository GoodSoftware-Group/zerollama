package voicepipe

import "testing"

func TestDuplexEnableMTP(t *testing.T) {
	if got := DuplexEnableMTP(false, "llama"); got != nil {
		t.Fatalf("no draft: %v", got)
	}
	got := DuplexEnableMTP(true, "llama")
	if got == nil || !*got {
		t.Fatalf("llama draft: %v", got)
	}
	got = DuplexEnableMTP(true, "qwen35")
	if got == nil || *got {
		t.Fatalf("qwen35: %v", got)
	}
	got = DuplexEnableMTP(true, "Qwen35MoE")
	if got == nil || *got {
		t.Fatalf("qwen35moe: %v", got)
	}
	got = DuplexEnableMTP(true, "qwen3")
	if got == nil || !*got {
		t.Fatalf("qwen3: %v", got)
	}
}
