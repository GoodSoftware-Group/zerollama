package server

import (
	"testing"

	"github.com/ollama/ollama/types/model"
)

func TestModelInferencePath(t *testing.T) {
	mlx := &Model{Config: model.ConfigV2{ModelFormat: "safetensors"}}
	if got := modelInferencePath(mlx); got != InferencePathMLX {
		t.Fatalf("mlx path = %q", got)
	}
	gguf := &Model{
		ModelPath: "/models/x.gguf",
		Config:    model.ConfigV2{ModelFormat: "gguf"},
	}
	if got := modelInferencePath(gguf); got != InferencePathGGUFGgml && got != InferencePathGGUFLlama {
		t.Fatalf("gguf path = %q", got)
	}
}

func TestGateSessionKeyPreservesGGUFClientKey(t *testing.T) {
	gguf := &Model{
		ModelPath: "/models/x.gguf",
		Config:    model.ConfigV2{ModelFormat: "gguf"},
	}
	raw := "ruby-trivia:bg:audit"
	got := gateSessionKey(gguf, "digest:abc", raw, mlxClassBackground, mlxQoS{CacheScope: qosCacheScopeShared})
	if got != raw {
		t.Fatalf("gateSessionKey(gguf) = %q want %q", got, raw)
	}
}

func TestGateSessionKeyUnkeyedGGUFStaysEmpty(t *testing.T) {
	gguf := &Model{
		ModelPath: "/models/x.gguf",
		Config:    model.ConfigV2{ModelFormat: "gguf"},
	}
	got := gateSessionKey(gguf, "digest:abc", "", mlxClassAuxiliary, mlxQoS{})
	if got != "" {
		t.Fatalf("unkeyed gguf must not invent aux/bg branch, got %q", got)
	}
}

func TestGateSessionKeyRewritesMLXBackground(t *testing.T) {
	mlx := &Model{Config: model.ConfigV2{ModelFormat: "safetensors"}}
	got := gateSessionKey(mlx, "digest:mlx", "", mlxClassBackground, mlxQoS{})
	if got == "" || got == "ruby-trivia:bg:audit" {
		t.Fatalf("expected mlx branch key, got %q", got)
	}
}

func TestModelSupportsSessionQoS(t *testing.T) {
	embed := &Model{
		ModelPath: "/models/e.gguf",
		Config: model.ConfigV2{
			ModelFormat:  "gguf",
			Capabilities: []string{string(model.CapabilityEmbedding)},
		},
	}
	if modelSupportsSessionQoS(embed) {
		t.Fatal("embedding-only model should not use session qos")
	}
	text := &Model{
		ModelPath: "/models/t.gguf",
		Config: model.ConfigV2{
			ModelFormat:  "gguf",
			Capabilities: []string{string(model.CapabilityCompletion)},
		},
	}
	if !modelSupportsSessionQoS(text) {
		t.Fatal("completion gguf should use session qos")
	}
}

func TestAgentSessionMetadataEnabled(t *testing.T) {
	if agentSessionMetadataEnabled(map[string]any{"prompt_cache_key": "custom-key"}) {
		t.Fatal("generic key should not enable eliza enrichment")
	}
	if !agentSessionMetadataEnabled(map[string]any{
		"prompt_cache_key": "hermes:agent:main:1",
	}) {
		t.Fatal("hermes key should enable eliza enrichment")
	}
	if !agentSessionMetadataEnabled(map[string]any{
		"prompt_cache_key": "x",
		"zerollama":        map[string]any{"qos_class": "background"},
	}) {
		t.Fatal("zerollama block should enable eliza enrichment")
	}
}
