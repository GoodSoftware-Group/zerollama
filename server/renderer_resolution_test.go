package server

import (
	"testing"

	"github.com/ollama/ollama/types/model"
)

func TestResolveRendererNameQwen38LibraryTag(t *testing.T) {
	got := resolveRendererName(&Model{Config: model.ConfigV2{
		ModelFamily: "qwen35",
		Renderer:    "qwen3.8",
	}})
	if got != "qwen3.8" {
		t.Fatalf("got %q, want qwen3.8", got)
	}
}

func TestResolveRendererNameQwen35Default(t *testing.T) {
	got := resolveRendererName(&Model{Config: model.ConfigV2{ModelFamily: "qwen35"}})
	if got != "qwen3.5" {
		t.Fatalf("got %q", got)
	}
}

func TestResolveRendererNameGptOSSDefault(t *testing.T) {
	got := resolveRendererName(&Model{Config: model.ConfigV2{ModelFamily: "gpt_oss"}})
	if got != "harmony" {
		t.Fatalf("got %q", got)
	}
}
