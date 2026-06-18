package server

import (
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/types/model"
)

func TestLlamaServerConfigForModel(t *testing.T) {
	tests := []struct {
		name string
		m    *Model
		opts api.Options
		want llm.LlamaServerConfig
	}{
		{
			name: "manifest draft defaults to draft-mtp",
			m:    &Model{DraftPath: "/models/draft.gguf"},
			want: llm.LlamaServerConfig{DraftModelPath: "/models/draft.gguf", SpecType: "draft-mtp"},
		},
		{
			name: "manifest draft with explicit eagle3 keeps eagle3",
			m:    &Model{DraftPath: "/models/drafter.gguf"},
			opts: api.Options{Runner: api.Runner{SpecType: "draft-eagle3"}},
			want: llm.LlamaServerConfig{DraftModelPath: "/models/drafter.gguf", SpecType: "draft-eagle3"},
		},
		{
			name: "sidecar draft_model_path defaults to draft-eagle3",
			m: &Model{
				Options: map[string]any{"draft_model_path": "/cache/drafter.gguf"},
			},
			want: llm.LlamaServerConfig{DraftModelPath: "/cache/drafter.gguf", SpecType: "draft-eagle3"},
		},
		{
			name: "manifest draft wins over sidecar option",
			m: &Model{
				DraftPath: "/models/manifest-draft.gguf",
				Options:   map[string]any{"draft_model_path": "/cache/drafter.gguf"},
			},
			want: llm.LlamaServerConfig{DraftModelPath: "/models/manifest-draft.gguf", SpecType: "draft-mtp"},
		},
		{
			name: "qwen35 without eliza prefix does not auto ngram",
			m: &Model{
				Name: "qwen3.6:latest",
				Config: model.ConfigV2{
					ModelFamily:   "qwen35moe",
					ModelFamilies: []string{"qwen35moe"},
				},
			},
			want: llm.LlamaServerConfig{},
		},
		{
			name: "eliza ngram requires env",
			m: &Model{
				Name: "eliza-1-2b:latest",
				Config: model.ConfigV2{
					ModelFamily:   "qwen35",
					ModelFamilies: []string{"qwen35"},
				},
			},
			want: llm.LlamaServerConfig{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := llamaServerConfigForModel(tt.m, false, tt.opts)
			if got.DraftModelPath != tt.want.DraftModelPath {
				t.Fatalf("DraftModelPath = %q, want %q", got.DraftModelPath, tt.want.DraftModelPath)
			}
			if got.SpecType != tt.want.SpecType {
				t.Fatalf("SpecType = %q, want %q", got.SpecType, tt.want.SpecType)
			}
		})
	}

	t.Run("eliza ngram when env enabled", func(t *testing.T) {
		t.Setenv("ZEROLLAMA_ELIZA_NGRAM", "1")
		if !envconfig.ElizaNgramDefault() {
			t.Fatal("expected ElizaNgramDefault")
		}
		got := llamaServerConfigForModel(&Model{
			Name: "eliza-1-27b-256k:latest",
			Config: model.ConfigV2{
				ModelFamily:   "qwen35",
				ModelFamilies: []string{"qwen35"},
			},
		}, false, api.Options{})
		if got.SpecType != "ngram-simple" {
			t.Fatalf("SpecType = %q, want ngram-simple", got.SpecType)
		}
	})
}
