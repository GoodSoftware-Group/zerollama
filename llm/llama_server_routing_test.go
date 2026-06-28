package llm

import (
	"testing"

	"github.com/ollama/ollama/envconfig"
)

func TestUseLlamaServerBackendRespectsExplicitOff(t *testing.T) {
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "0")
	if useLlamaServerBackendForModelGOOS("linux", nil, true, LlamaServerConfig{}) {
		t.Fatal("ZEROLLAMA_LLAMA_SERVER=0 must disable llama-server routing")
	}
}

func TestUseLlamaServerBackendExplicitOn(t *testing.T) {
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "1")
	if !useLlamaServerBackendForModelGOOS("linux", nil, true, LlamaServerConfig{}) {
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
	if !useLlamaServerBackendForModelGOOS("linux", nil, true, LlamaServerConfig{}) {
		t.Fatal("Linux auto must route plain text GGUF through llama-server")
	}
}

func TestUseLlamaServerBackendLinuxAutoVision(t *testing.T) {
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "auto")
	if !useLlamaServerBackendForModelGOOS("linux", []string{"/path/to/mmproj.gguf"}, true, LlamaServerConfig{}) {
		t.Fatal("Linux auto must route vision GGUF through llama-server (upstream parity)")
	}
}

func TestUseLlamaServerBackendAutoNotOnDarwin(t *testing.T) {
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "auto")
	if useLlamaServerBackendForModelGOOS("darwin", nil, true, LlamaServerConfig{}) {
		t.Fatal("Linux auto value must not enable llama-server on Darwin for plain GGUF")
	}
}

func TestUseLlamaServerBackendDarwinSpecAuto(t *testing.T) {
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "")
	cfg := LlamaServerConfig{SpecType: "draft-eagle3", DraftModelPath: "/cache/drafter.gguf"}
	if !useLlamaServerBackendForModelGOOS("darwin", nil, true, cfg) {
		t.Fatal("Darwin should auto-route speculative models when llama-server is discoverable")
	}
}

func TestUseLlamaServerBackendDarwinPlainGGUF(t *testing.T) {
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "")
	if useLlamaServerBackendForModelGOOS("darwin", nil, true, LlamaServerConfig{}) {
		t.Fatal("plain GGUF should stay on ggml Metal on Darwin")
	}
}

func TestUseLlamaServerBackendDarwinSpecNeedsDiscoverable(t *testing.T) {
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "0")
	cfg := LlamaServerConfig{SpecType: "draft-eagle3"}
	if useLlamaServerBackendForModelGOOS("darwin", nil, false, cfg) {
		t.Fatal("spec auto-route requires discoverable llama-server")
	}
}

func TestModelNeedsLlamaServerSpec(t *testing.T) {
	if !ModelNeedsLlamaServerSpec(LlamaServerConfig{SpecType: "ngram-simple"}) {
		t.Fatal("ngram")
	}
	if !ModelNeedsLlamaServerSpec(LlamaServerConfig{DraftModelPath: "/d.gguf"}) {
		t.Fatal("draft path")
	}
	if ModelNeedsLlamaServerSpec(LlamaServerConfig{}) {
		t.Fatal("plain")
	}
}

func TestSpecModelRequiresLlamaServerError(t *testing.T) {
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "0")
	cfg := LlamaServerConfig{SpecType: "draft-eagle3"}
	if err := SpecModelRequiresLlamaServerError(cfg); err == nil {
		t.Fatal("expected error when spec model and llama-server disabled")
	}
}

func TestUseLlamaServerBackendAutoRequiresDiscoverable(t *testing.T) {
	t.Setenv("ZEROLLAMA_LLAMA_SERVER", "auto")
	if useLlamaServerBackendForModelGOOS("linux", nil, false, LlamaServerConfig{}) {
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
