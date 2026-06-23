package server

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
)

func TestPrimaryModelFamilyPrefersLLMOverClip(t *testing.T) {
	cfg := model.ConfigV2{
		ModelFamily:   "clip",
		ModelFamilies: []string{"clip", "qwen35"},
	}
	if got := primaryModelFamily(cfg); got != "qwen35" {
		t.Fatalf("primaryModelFamily() = %q, want qwen35", got)
	}
}

func TestPrimaryModelFamilyQwen35Moe(t *testing.T) {
	cfg := model.ConfigV2{
		ModelFamily:   "clip",
		ModelFamilies: []string{"clip", "qwen35moe"},
	}
	if got := primaryModelFamily(cfg); got != "qwen35moe" {
		t.Fatalf("primaryModelFamily() = %q, want qwen35moe", got)
	}
}

func TestResolveParserNameForQwen35VLManifest(t *testing.T) {
	m := &Model{Config: model.ConfigV2{
		ModelFamily:   "clip",
		ModelFamilies: []string{"clip", "qwen35"},
	}}
	if got := resolveParserName(m); got != "qwen3.5" {
		t.Fatalf("resolveParserName() = %q, want qwen3.5", got)
	}
}

func TestDefaultRendererForBrokenVLManifest(t *testing.T) {
	m := &Model{Config: model.ConfigV2{
		ModelFamily:   "clip",
		ModelFamilies: []string{"clip", "qwen35"},
	}}
	if got := defaultRendererForFamily(m); got != "qwen3.5" {
		t.Fatalf("defaultRendererForFamily() = %q, want qwen3.5", got)
	}
}

func TestPrimaryModelFamilyProjectorOnlyReturnsEmpty(t *testing.T) {
	cfg := model.ConfigV2{
		ModelFamily:   "clip",
		ModelFamilies: []string{"clip"},
	}
	if got := primaryModelFamily(cfg); got != "" {
		t.Fatalf("primaryModelFamily() = %q, want empty for projector-only", got)
	}
}

func TestPrimaryModelFamilyFromModelFamilyOnlyClip(t *testing.T) {
	cfg := model.ConfigV2{
		ModelFamily: "clip",
	}
	if got := primaryModelFamily(cfg); got != "" {
		t.Fatalf("primaryModelFamily() = %q, want empty when only ModelFamily is clip", got)
	}
}

func TestIsGptOSSFamily(t *testing.T) {
	for _, arch := range []string{"gpt_oss", "gptoss", "gpt-oss", "GPT_OSS"} {
		if !isGptOSSFamily(arch) {
			t.Fatalf("isGptOSSFamily(%q) = false, want true", arch)
		}
	}
	if isGptOSSFamily("llama3") {
		t.Fatal("isGptOSSFamily(llama3) = true, want false")
	}
}

func TestResolveParserNameForGptOSSMLX(t *testing.T) {
	m := &Model{Config: model.ConfigV2{ModelFamily: "gpt_oss", ModelFormat: "safetensors"}}
	if got := resolveParserName(m); got != "harmony" {
		t.Fatalf("resolveParserName() = %q, want harmony", got)
	}
}

func TestCapabilitiesGptOSSMLX(t *testing.T) {
	tmpl, err := template.Parse("{{ .Prompt }}")
	if err != nil {
		t.Fatal(err)
	}
	m := Model{
		Config: model.ConfigV2{
			ModelFamily: "gpt_oss",
			ModelFormat: "safetensors",
			Capabilities: []string{"completion"},
		},
		Template: tmpl,
	}
	caps := m.Capabilities()
	for _, want := range []model.Capability{model.CapabilityTools, model.CapabilityThinking} {
		if !slices.Contains(caps, want) {
			t.Fatalf("Capabilities() missing %q, got %v", want, caps)
		}
	}
	if err := m.CheckCapabilities(model.CapabilityTools, model.CapabilityThinking); err != nil {
		t.Fatalf("CheckCapabilities() = %v", err)
	}
}
