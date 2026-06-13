package server

import (
	"testing"

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
