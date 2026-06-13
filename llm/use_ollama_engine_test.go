package llm

import "testing"

func TestPickOllamaEngineDarwinQwen35Legacy(t *testing.T) {
	for _, arch := range []string{"qwen35", "qwen35moe", "qwen3next"} {
		if pickOllamaEngine(arch, false, true, true) {
			t.Fatalf("darwin should use legacy llamarunner for %q until Go Metal reserve is stable", arch)
		}
	}
	if !pickOllamaEngine("qwen3", false, true, true) {
		t.Fatal("qwen3 should use Go engine on darwin")
	}
}

func TestPickOllamaEngineNewEngineOverride(t *testing.T) {
	if !pickOllamaEngine("qwen35moe", true, true, true) {
		t.Fatal("OLLAMA_NEW_ENGINE=1 should force Go engine (debug; may abort on qwen35 load)")
	}
}

func TestPickOllamaEngineNotRequired(t *testing.T) {
	if pickOllamaEngine("llama", false, false, true) {
		t.Fatal("architectures not in OllamaEngineRequired should use legacy runner")
	}
}
