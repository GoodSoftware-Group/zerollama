package renderers

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

// Qwen3-VL uses one <|vision_start|><|image_pad|><|vision_end|> per frame (SGLang parity).
// Production chat sets RenderImgTags=true → [img-N] per raster for the flat runner list.
func TestQwen3VLRenderer_videoSpansVisionTokens(t *testing.T) {
	r := &Qwen3VLRenderer{isThinking: false, useImgTags: false}
	msg := api.Message{
		Role:    "user",
		Content: "clip",
		Images:  make([]api.ImageData, 4),
		VideoSpans: []api.VideoSpan{
			{FrameCount: 3},
		},
	}
	got, _ := r.renderContent([]api.Message{msg}, 0, 0)
	vision := "<|vision_start|><|image_pad|><|vision_end|>"
	wantPrefix := strings.Repeat(vision, 4) // 1 still + 3 video frames
	if !strings.HasPrefix(got, wantPrefix) {
		t.Fatalf("prefix=%q want 4 vision blocks (1 still + 3 video frames)", got)
	}
	if !strings.HasSuffix(got, "clip") {
		t.Fatalf("missing content suffix: %q", got)
	}
}

func TestQwen3VLRenderer_videoSpansUseImgTagsPerFrame(t *testing.T) {
	r := &Qwen3VLRenderer{isThinking: false, useImgTags: true}
	msg := api.Message{
		Role:    "user",
		Content: "clip",
		Images:  make([]api.ImageData, 4),
		VideoSpans: []api.VideoSpan{
			{FrameCount: 3},
		},
	}
	got, next := r.renderContent([]api.Message{msg}, 0, 0)
	if !strings.HasPrefix(got, "[img-0][img-1][img-2][img-3]") {
		t.Fatalf("prefix=%q want per-frame img tags", got)
	}
	if next != 4 {
		t.Fatalf("next offset=%d want 4", next)
	}
}

func TestQwen3VLRenderer_skipsImgTagsWhenPaddedInputIDs(t *testing.T) {
	r := &Qwen3VLRenderer{isThinking: false, useImgTags: true}
	msg := api.Message{
		Role:           "user",
		Content:        "clip",
		Images:         make([]api.ImageData, 3),
		VideoSpans:     []api.VideoSpan{{FrameCount: 3}},
		PaddedInputIDs: []int{1, 2, 3},
	}
	got, next := r.renderContent([]api.Message{msg}, 0, 0)
	if strings.Contains(got, "[img-") {
		t.Fatalf("expected no img tags with padded_input_ids, got %q", got)
	}
	if !strings.HasSuffix(got, "clip") {
		t.Fatalf("missing content: %q", got)
	}
	if next != 0 {
		t.Fatalf("image offset=%d want 0", next)
	}
}

func TestQwen3VLRenderer_skipsVisionWhenPaddedInputIDs(t *testing.T) {
	r := &Qwen3VLRenderer{isThinking: false, useImgTags: false}
	msg := api.Message{
		Role:           "user",
		Content:        "clip",
		Images:         make([]api.ImageData, 3),
		VideoSpans:     []api.VideoSpan{{FrameCount: 3}},
		PaddedInputIDs: []int{1, 2, 3, 4, 5},
	}
	got, next := r.renderContent([]api.Message{msg}, 0, 0)
	if strings.Contains(got, "<|vision_start|>") {
		t.Fatalf("expected no vision blocks with padded_input_ids, got %q", got)
	}
	if !strings.HasSuffix(got, "clip") {
		t.Fatalf("missing content: %q", got)
	}
	if next != 0 {
		t.Fatalf("image offset=%d want 0", next)
	}
}

func TestQwen3VLRenderer_toolResultImages(t *testing.T) {
	// SGLang #33898: tool messages must emit vision placeholders from renderContent.
	r := &Qwen3VLRenderer{isThinking: false, useImgTags: true}
	msgs := []api.Message{
		{Role: "user", Content: "look"},
		{Role: "assistant", Content: "", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "see"}}}},
		{Role: "tool", Content: "shot", Images: []api.ImageData{[]byte{1, 2, 3}}},
	}
	got, err := r.Render(msgs, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, "<tool_response>\n[img-0]shot\n</tool_response>") {
		t.Fatalf("tool response missing vision placeholder: %q", got)
	}
}
