package model

import (
	"testing"

	modeltypes "github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/imagegen/manifest"
)

func TestReadDraftConfigKnownLayouts(t *testing.T) {
	t.Parallel()
	jsonType := "application/vnd.ollama.image.json"

	tests := []struct {
		name   string
		layers []manifest.ManifestLayer
		want   *modeltypes.Draft
	}{
		{
			name: "draft dir",
			layers: []manifest.ManifestLayer{
				{MediaType: jsonType, Name: "draft/config.json", Digest: "sha256:a"},
			},
			want: &modeltypes.Draft{ModelFormat: "safetensors", TensorPrefix: "draft.", Config: "draft/config.json"},
		},
		{
			name: "drafter dir mlx-serve",
			layers: []manifest.ManifestLayer{
				{MediaType: jsonType, Name: "drafter/config.json", Digest: "sha256:b"},
			},
			want: &modeltypes.Draft{ModelFormat: "safetensors", TensorPrefix: "drafter.", Config: "drafter/config.json"},
		},
		{
			name: "assistant dir",
			layers: []manifest.ManifestLayer{
				{MediaType: jsonType, Name: "assistant/config.json", Digest: "sha256:c"},
			},
			want: &modeltypes.Draft{ModelFormat: "safetensors", TensorPrefix: "assistant.", Config: "assistant/config.json"},
		},
		{
			name: "mtp dir mlx-serve qwen",
			layers: []manifest.ManifestLayer{
				{MediaType: jsonType, Name: "mtp/config.json", Digest: "sha256:d"},
			},
			want: &modeltypes.Draft{ModelFormat: "safetensors", TensorPrefix: "mtp.", Config: "mtp/config.json"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &manifest.ModelManifest{
				Manifest: &manifest.Manifest{
					Config: manifest.ManifestLayer{Digest: "sha256:cfg"},
					Layers: tt.layers,
				},
			}
			got := readDraftConfig(m)
			if tt.want == nil {
				if got != nil {
					t.Fatalf("got %+v, want nil", got)
				}
				return
			}
			if got == nil {
				t.Fatal("got nil")
			}
			if got.Config != tt.want.Config || got.TensorPrefix != tt.want.TensorPrefix {
				t.Fatalf("got %+v, want %+v", got, tt.want)
			}
		})
	}
}
