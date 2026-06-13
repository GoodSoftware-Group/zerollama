package llm

import "testing"

func TestPickOllamaEngineRequired(t *testing.T) {
	for _, arch := range []string{"qwen35", "qwen35moe", "qwen3next", "qwen3"} {
		if !pickOllamaEngine(arch, false, true) {
			t.Fatalf("OllamaEngineRequired arch %q should use Go engine", arch)
		}
	}
}

func TestPickOllamaEngineNewEngineOverride(t *testing.T) {
	if !pickOllamaEngine("llama", true, false) {
		t.Fatal("OLLAMA_NEW_ENGINE=1 should force Go engine")
	}
}

func TestPickOllamaEngineNotRequired(t *testing.T) {
	if pickOllamaEngine("llama", false, false) {
		t.Fatal("architectures not in OllamaEngineRequired should use legacy runner")
	}
}
