package create

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestGemma4EmbeddingManifestCaps(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want []string
	}{
		{
			name: "text only",
			raw:  `{"architectures":["EmbeddingGemma2Model"],"text_config":{"embedding_dim":768,"max_position_embeddings":262144}}`,
			want: []string{"embedding"},
		},
		{
			name: "vision tower",
			raw:  `{"architectures":["EmbeddingGemma2Model"],"text_config":{"embedding_dim":768,"max_position_embeddings":262144},"vision_config":{"model_type":"gemma4_vision"}}`,
			want: []string{"embedding", "vision"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(tt.raw), 0o644); err != nil {
				t.Fatal(err)
			}
			caps, embedLen, ctxLen, err := EmbeddingManifestCaps(dir)
			if err != nil {
				t.Fatal(err)
			}
			if embedLen != 768 {
				t.Errorf("embedLen = %d, want 768", embedLen)
			}
			if ctxLen != 8192 {
				t.Errorf("ctxLen = %d, want 8192", ctxLen)
			}
			if !slices.Equal(caps, tt.want) {
				t.Errorf("caps = %v, want %v", caps, tt.want)
			}
		})
	}
}

func TestGemma4EmbeddingProjectionExcluded(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"text_config":{"embedding_dim":768,"max_position_embeddings":8192}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	tr, err := newGemma4EmbeddingImportTransform(dir, sourceModelConfig{})
	if err != nil {
		t.Fatal(err)
	}
	p := tr.(gemma4EmbeddingImportTransform)
	shape := []int32{768, 512}
	for _, name := range []string{
		"embedding_projection.weight",
		"model.embedding_projection.weight",
		"language_model.embedding_projection.weight",
	} {
		if got := p.quantizationType(name, shape, "nvfp4"); got != "" {
			t.Errorf("quantizationType(%q) = %q, want \"\"", name, got)
		}
	}
	if got := p.quantizationType("language_model.layers.0.mlp.gate_proj.weight", []int32{2048, 512}, "nvfp4"); got != "nvfp4" {
		t.Errorf("trunk tensor quantization = %q, want nvfp4", got)
	}
}
