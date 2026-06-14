package llm

import "testing"

func TestPlainTextGGUFEligibleForLlamaServer(t *testing.T) {
	if !plainTextGGUFEligibleForLlamaServer(nil) {
		t.Fatal("no projectors should be eligible")
	}
	if plainTextGGUFEligibleForLlamaServer([]string{"/path/to/mmproj.gguf"}) {
		t.Fatal("split vision projector should stay on legacy runner until parity")
	}
}

func TestUseLlamaServerBackendRespectsExplicitOff(t *testing.T) {
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "0")
	if useLlamaServerBackend(nil) {
		t.Fatal("ZEROLLAMA_LLAMA_SERVER=0 must disable llama-server routing")
	}
}

func TestUseLlamaServerBackendExplicitOn(t *testing.T) {
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "1")
	if !useLlamaServerBackend(nil) {
		t.Fatal("ZEROLLAMA_LLAMA_SERVER=1 must enable llama-server routing")
	}
}

func TestUseLlamaServerBackendRejectsProjectorsEvenWhenExplicit(t *testing.T) {
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "1")
	if useLlamaServerBackend([]string{"/path/to/mmproj.gguf"}) {
		t.Fatal("vision projector must stay on legacy runner even when ZEROLLAMA_LLAMA_SERVER=1")
	}
}
