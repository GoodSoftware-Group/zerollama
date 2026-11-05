package modelrepair

import "testing"

func TestSuggestedFamilyChatStack(t *testing.T) {
	r, p, ok := suggestedFamilyChatStack(ShowInfo{Architecture: "deepseek4"})
	if !ok || r != "deepseek3" || p != "deepseek3" {
		t.Fatalf("deepseek4: %q %q ok=%v", r, p, ok)
	}
	r, p, ok = suggestedFamilyChatStack(ShowInfo{Architecture: "qwen35moe"})
	if !ok || r != "qwen3.5" || p != "qwen3.5" {
		t.Fatalf("qwen35moe: %q %q ok=%v", r, p, ok)
	}
	_, _, ok = suggestedFamilyChatStack(ShowInfo{Architecture: "llama4"})
	if ok {
		t.Fatal("llama4 has no auto renderer stack yet")
	}
}
