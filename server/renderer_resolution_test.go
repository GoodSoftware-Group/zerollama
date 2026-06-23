package server

import (
	"testing"

	"github.com/ollama/ollama/types/model"
)

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
