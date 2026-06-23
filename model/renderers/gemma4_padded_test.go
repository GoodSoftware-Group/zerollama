package renderers

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestGemma4Renderer_skipsImgTagsWhenPaddedInputIDs(t *testing.T) {
	r := &Gemma4Renderer{useImgTags: true}
	msgs := []api.Message{{
		Role:           "user",
		Content:        "clip",
		Images:         []api.ImageData{{1}, {2}},
		PaddedInputIDs: []int{100, 101},
	}}
	var sb strings.Builder
	offset := 0
	r.renderContent(&sb, msgs, 0, msgs[0], &offset, true)
	if got := sb.String(); got != "clip" {
		t.Fatalf("got %q want content only", got)
	}
	if offset != 0 {
		t.Fatalf("offset=%d want 0", offset)
	}
}

func TestGemma4Renderer_skipsHFImageWhenPaddedInputIDs(t *testing.T) {
	r := &Gemma4Renderer{useImgTags: false}
	msgs := []api.Message{{
		Role:           "user",
		Content:        "pic",
		Images:         []api.ImageData{{1}},
		PaddedInputIDs: []int{200},
	}}
	var sb strings.Builder
	offset := 0
	r.renderContent(&sb, msgs, 0, msgs[0], &offset, true)
	if strings.Contains(sb.String(), "<|image|>") {
		t.Fatalf("should skip HF image placeholder: %q", sb.String())
	}
}
