package server

import (
	"testing"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
)

func TestDeferInferenceToRuntime(t *testing.T) {
	m := &Model{
		ModelPath: "/tmp/runtime.gguf",
		Config: model.ConfigV2{
			Capabilities: []string{string(model.CapabilityCompletion)},
			ModalityBackends: map[string]string{
				model.ModalityInference: model.BackendZerollamaRuntime,
			},
		},
	}

	t.Setenv("ZEROLLAMA_RUNTIME_URL", "")
	t.Setenv("ZEROLLAMA_LEGACY_RUNNER", "")
	if deferInferenceToRuntime(m) {
		t.Fatal("expected false without runtime URL")
	}

	t.Setenv("ZEROLLAMA_RUNTIME_URL", "http://127.0.0.1:8081")
	if !deferInferenceToRuntime(m) {
		t.Fatal("expected true for zerollama-runtime model")
	}

	t.Setenv("ZEROLLAMA_LEGACY_RUNNER", "1")
	if deferInferenceToRuntime(m) {
		t.Fatal("expected false with legacy runner forced")
	}
}

func TestRuntimeDefaultOnImplicit(t *testing.T) {
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "http://127.0.0.1:8081")
	t.Setenv("ZEROLLAMA_RUNTIME", "")
	if !envconfig.RuntimeDefaultOn() {
		t.Fatal("unset ZEROLLAMA_RUNTIME should default on when URL set")
	}
	t.Setenv("ZEROLLAMA_RUNTIME", "0")
	if envconfig.RuntimeDefaultOn() {
		t.Fatal("ZEROLLAMA_RUNTIME=0 should disable")
	}
}
