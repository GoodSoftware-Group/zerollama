package llm

import (
	"testing"

	"github.com/ollama/ollama/envconfig"
)

func TestUseLlamaServerBackendRespectsExplicitOff(t *testing.T) {
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "0")
	if useLlamaServerBackendGOOS("linux", nil, true) {
		t.Fatal("ZEROLLAMA_LLAMA_SERVER=0 must disable llama-server routing")
	}
}

func TestUseLlamaServerBackendExplicitOn(t *testing.T) {
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "1")
	if !useLlamaServerBackendGOOS("linux", nil, true) {
		t.Fatal("ZEROLLAMA_LLAMA_SERVER=1 must enable llama-server routing")
	}
}

func TestUseLlamaServerBackendExplicitOnWithProjectors(t *testing.T) {
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "1")
	if !useLlamaServerBackend([]string{"/path/to/mmproj.gguf"}) {
		t.Fatal("ZEROLLAMA_LLAMA_SERVER=1 must route vision GGUF through llama-server (upstream parity)")
	}
}

func TestUseLlamaServerBackendRejectsProjectorsWithoutExplicit(t *testing.T) {
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "0")
	if useLlamaServerBackend([]string{"/path/to/mmproj.gguf"}) {
		t.Fatal("vision projector must stay on legacy runner when llama-server disabled")
	}
}

func TestUseLlamaServerBackendLinuxAutoPlainText(t *testing.T) {
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "auto")
	if !useLlamaServerBackendGOOS("linux", nil, true) {
		t.Fatal("Linux auto must route plain text GGUF through llama-server")
	}
}

func TestUseLlamaServerBackendLinuxAutoVision(t *testing.T) {
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "auto")
	if !useLlamaServerBackendGOOS("linux", []string{"/path/to/mmproj.gguf"}, true) {
		t.Fatal("Linux auto must route vision GGUF through llama-server (upstream parity)")
	}
}

func TestUseLlamaServerBackendAutoNotOnDarwin(t *testing.T) {
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "auto")
	if useLlamaServerBackendGOOS("darwin", nil, true) {
		t.Fatal("Linux auto value must not enable llama-server on Darwin")
	}
}

func TestUseLlamaServerBackendAutoRequiresDiscoverable(t *testing.T) {
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "auto")
	if useLlamaServerBackendGOOS("linux", nil, false) {
		t.Fatal("Linux auto must require discoverable llama-server binary")
	}
}

func TestLlamaServerBackendAutoEnv(t *testing.T) {
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "auto")
	if !envconfig.LlamaServerBackend() {
		t.Fatal("auto should count as llama-server backend enabled")
	}
	if envconfig.LlamaServerBackendExplicit() {
		t.Fatal("auto should not count as explicit opt-in")
	}
	if !envconfig.LlamaServerBackendAuto() {
		t.Fatal("auto should be detected")
	}
}
