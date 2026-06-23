package renderers

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestGemma4Renderer_videoSpansHFPlaceholders(t *testing.T) {
	r := &Gemma4Renderer{useImgTags: false}
	var sb strings.Builder
	offset := 0
	msg := api.Message{
		Role:    "user",
		Content: "what happens?",
		Images:  make([]api.ImageData, 5),
		VideoSpans: []api.VideoSpan{
			{FrameCount: 3},
		},
	}
	r.renderContent(&sb, []api.Message{msg}, 0, msg, &offset, true)
	got := sb.String()
	if !strings.HasPrefix(got, "<|image|><|image|><|video|>") {
		t.Fatalf("prefix=%q, want still images then video token", got)
	}
	if !strings.HasSuffix(got, "what happens?") {
		t.Fatalf("suffix missing content: %q", got)
	}
}

func TestGemma4Renderer_stillImagesOnlyHFPlaceholders(t *testing.T) {
	r := &Gemma4Renderer{useImgTags: false}
	var sb strings.Builder
	offset := 0
	msg := api.Message{
		Role:    "user",
		Content: "pic",
		Images:  []api.ImageData{{1}, {2}},
	}
	r.renderContent(&sb, []api.Message{msg}, 0, msg, &offset, true)
	if got := sb.String(); got != "<|image|><|image|>pic" {
		t.Fatalf("got %q", got)
	}
}

func TestGemma4Renderer_videoSpansUseImgTagsPerFrame(t *testing.T) {
	r := &Gemma4Renderer{useImgTags: true}
	var sb strings.Builder
	offset := 0
	msg := api.Message{
		Role:    "user",
		Content: "clip",
		Images:  make([]api.ImageData, 5),
		VideoSpans: []api.VideoSpan{
			{FrameCount: 3},
		},
	}
	r.renderContent(&sb, []api.Message{msg}, 0, msg, &offset, true)
	got := sb.String()
	if !strings.HasPrefix(got, "[img-0][img-1][img-2][img-3][img-4]") {
		t.Fatalf("prefix=%q, want per-frame img tags", got)
	}
	if offset != 5 {
		t.Fatalf("offset=%d, want 5", offset)
	}
}

func TestGemma4Renderer_audioClipsHFPlaceholders(t *testing.T) {
	r := &Gemma4Renderer{useImgTags: false}
	var sb strings.Builder
	offset := 0
	msg := api.Message{
		Role:       "user",
		Content:    "listen",
		AudioClips: []api.AudioData{{1}, {2}},
	}
	r.renderContent(&sb, []api.Message{msg}, 0, msg, &offset, true)
	if got := sb.String(); got != "<|audio|><|audio|>listen" {
		t.Fatalf("got %q", got)
	}
}
