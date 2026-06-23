package llm

import "testing"

func TestPickOllamaEngineRequired(t *testing.T) {
	for _, arch := range []string{"qwen35", "qwen35moe", "qwen3next", "qwen3"} {
		if !pickOllamaEngine(true) {
			t.Fatalf("OllamaEngineRequired arch %q should use Go engine", arch)
		}
	}
}

func TestPickOllamaEngineNotRequired(t *testing.T) {
	if pickOllamaEngine(false) {
		t.Fatal("architectures not in OllamaEngineRequired should use legacy runner")
	}
}
