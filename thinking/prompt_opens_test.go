package thinking

import "testing"

func TestPromptOpensThink(t *testing.T) {
	tests := []struct {
		in   string
		want bool
	}{
		{"", false},
		{"<|im_start|>assistant\n", false},
		{"<|im_start|>assistant\n<think>", true},
		{"<|im_start|>assistant\n<think>\n", true},
		{"<think>old</think>ok", false},
		{"<think>old</think>\n<think>", true},
		{"</think>answer", false},
	}
	for _, tt := range tests {
		if got := PromptOpensThink(tt.in); got != tt.want {
			t.Fatalf("PromptOpensThink(%q)=%v want %v", tt.in, got, tt.want)
		}
	}
}

func TestParserSeedFromPrompt(t *testing.T) {
	p := &Parser{OpeningTag: "<think>", ClosingTag: "</think>"}
	p.SeedFromPrompt("<|im_start|>assistant\n<think>\n")
	thinking, content := p.AddContent("plan</think>Answer")
	if thinking != "plan" || content != "Answer" {
		t.Fatalf("thinking=%q content=%q", thinking, content)
	}
}
