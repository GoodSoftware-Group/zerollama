package server

import (
	"context"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
)

func TestChatPrompt_attachesGridTHWPerVideoFrame(t *testing.T) {
	tmpl, err := template.Parse("{{ range .Messages }}{{ .Role }}: {{ .Content }}{{ end }}")
	if err != nil {
		t.Fatal(err)
	}
	m := &Model{
		Template:       tmpl,
		ProjectorPaths: []string{"mmproj"},
		Config:         model.ConfigV2{ModelFamilies: []string{"qwen3vl"}, Renderer: "qwen3-vl-instruct"},
	}
	msgs := []api.Message{{
		Role:    "user",
		Content: "describe",
		Images:  make([]api.ImageData, 4),
		VideoSpans: []api.VideoSpan{{
			FrameCount: 4,
			GridTHW:    []int{4, 24, 32},
		}},
	}}
	_, images, _, _, err := chatPrompt(context.Background(), m, nil, &api.Options{}, msgs, nil, nil, false, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(images) != 4 {
		t.Fatalf("images=%d want 4", len(images))
	}
	for i, img := range images {
		if len(img.GridTHW) != 3 {
			t.Fatalf("image %d grid=%v want [1,24,32]", i, img.GridTHW)
		}
		if img.GridTHW[0] != 1 || img.GridTHW[1] != 24 || img.GridTHW[2] != 32 {
			t.Fatalf("image %d grid=%v", i, img.GridTHW)
		}
	}
}
