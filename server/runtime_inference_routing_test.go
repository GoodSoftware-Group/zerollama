package server

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
)

func TestModelEligibleForRuntimeDefault(t *testing.T) {
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "http://127.0.0.1:8081")
	t.Setenv("ZEROLLAMA_LEGACY_RUNNER", "")

	text := &Model{
		ModelPath: "/tmp/m.gguf",
		Config: model.ConfigV2{
			Capabilities: []string{string(model.CapabilityCompletion)},
		},
	}
	if !modelEligibleForRuntimeDefault(text) {
		t.Fatal("expected text GGUF eligible")
	}
	if !modelUsesRuntimeInference(text) {
		t.Fatal("expected runtime default on")
	}

	embed := &Model{
		ModelPath: "/tmp/e.gguf",
		Config: model.ConfigV2{
			Capabilities: []string{string(model.CapabilityEmbedding)},
		},
	}
	if modelEligibleForRuntimeDefault(embed) {
		t.Fatal("embedding model should not be eligible")
	}

	vision := &Model{
		ModelPath: "/tmp/v.gguf",
		Config: model.ConfigV2{
			Capabilities: []string{
				string(model.CapabilityCompletion),
				string(model.CapabilityVision),
			},
		},
	}
	if modelEligibleForRuntimeDefault(vision) {
		t.Fatal("vision model should not be eligible")
	}

	explicit := &Model{
		ModelPath: "/tmp/x.gguf",
		Config: model.ConfigV2{
			Capabilities: []string{string(model.CapabilityCompletion)},
			ModalityBackends: map[string]string{
				model.ModalityInference: model.BackendZerollamaRuntime,
			},
		},
	}
	if !modelUsesRuntimeInference(explicit) {
		t.Fatal("explicit zerollama-runtime should use runtime")
	}

	tools := &Model{
		ModelPath: "/tmp/tools.gguf",
		Config: model.ConfigV2{
			Capabilities: []string{
				string(model.CapabilityCompletion),
				string(model.CapabilityTools),
			},
		},
	}
	if !modelEligibleForRuntimeDefault(tools) {
		t.Fatal("tools-capable text GGUF should be runtime-default eligible")
	}
	if !modelUsesRuntimeInference(tools) {
		t.Fatal("default-on should route tools-capable models to runtime")
	}
}

func TestDeferInferenceToRuntimeWithToolsBackend(t *testing.T) {
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "http://127.0.0.1:8081")
	t.Setenv("ZEROLLAMA_LEGACY_RUNNER", "")

	toolsExplicit := &Model{
		ModelPath: "/tmp/tools.gguf",
		Config: model.ConfigV2{
			Capabilities: []string{
				string(model.CapabilityCompletion),
				string(model.CapabilityTools),
			},
			ModalityBackends: map[string]string{
				model.ModalityInference: model.BackendZerollamaRuntime,
			},
		},
	}
	if !modelUsesRuntimeInference(toolsExplicit) {
		t.Fatal("explicit runtime backend should proxy")
	}
	if !deferInferenceToRuntime(toolsExplicit) {
		t.Fatal("explicit runtime backend should defer ggml load for tools models")
	}

	plain := &Model{
		ModelPath: "/tmp/plain.gguf",
		Config: model.ConfigV2{
			Capabilities: []string{string(model.CapabilityCompletion)},
			ModalityBackends: map[string]string{
				model.ModalityInference: model.BackendZerollamaRuntime,
			},
		},
	}
	if !deferInferenceToRuntime(plain) {
		t.Fatal("plain text + explicit runtime should defer ggml load")
	}
}

func TestRuntimeProxyActiveNotDefaultOn(t *testing.T) {
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "http://127.0.0.1:8081")
	t.Setenv("ZEROLLAMA_RUNTIME", "1")
	if !envconfig.RuntimeDefaultOn() {
		t.Fatal("expected default on")
	}
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodGet, "/", nil)
	if runtimeProxyActive(c) {
		t.Fatal("runtimeProxyActive should be false without ALL or header")
	}
}
