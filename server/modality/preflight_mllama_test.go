package modality

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestPreflightMllamaSingleImage_skipsNonMllama(t *testing.T) {
	policy := VideoSamplingPolicy{MaxFrames: 8}
	req := &api.ChatRequest{
		Messages: []api.Message{{Videos: []api.VideoData{{1}, {2}}}},
	}
	if err := PreflightMllamaSingleImage(policy, []string{"qwen3vl"}, req); err != nil {
		t.Fatal(err)
	}
}

func TestPreflightMllamaSingleImage_rejectsMultiFrameVideo(t *testing.T) {
	policy := VideoSamplingPolicy{MaxFrames: 4}
	req := &api.ChatRequest{
		Messages: []api.Message{{Videos: []api.VideoData{{9}}}},
	}
	if err := PreflightMllamaSingleImage(policy, []string{"mllama"}, req); err == nil {
		t.Fatal("expected error for video that expands to 4 frames")
	}
}

func TestPreflightMllamaSingleImage_allowsSingleStill(t *testing.T) {
	policy := VideoSamplingPolicy{MaxFrames: 8}
	req := &api.ChatRequest{
		Messages: []api.Message{{Images: []api.ImageData{{1}}}},
	}
	if err := PreflightMllamaSingleImage(policy, []string{"mllama"}, req); err != nil {
		t.Fatal(err)
	}
}

func TestPreflightMllamaSingleImage_rejectsPreexpandedMultiFrame(t *testing.T) {
	policy := VideoSamplingPolicy{MaxFrames: 8}
	req := &api.ChatRequest{
		Messages: []api.Message{{
			Role:       "user",
			Images:     make([]api.ImageData, 3),
			VideoSpans: []api.VideoSpan{{FrameCount: 3}},
		}},
	}
	if err := PreflightMllamaSingleImage(policy, []string{"mllama"}, req); err == nil {
		t.Fatal("expected error for pre-expanded multi-frame turn")
	}
}

func TestPreflightMllamaSingleImage_ignoresHistoricalPreexpanded(t *testing.T) {
	policy := VideoSamplingPolicy{MaxFrames: 8}
	req := &api.ChatRequest{
		Messages: []api.Message{
			{
				Role:       "user",
				Images:     make([]api.ImageData, 4),
				VideoSpans: []api.VideoSpan{{FrameCount: 4}},
			},
			{Role: "assistant", Content: "ok"},
			{Role: "user", Content: "follow up"},
		},
	}
	if err := PreflightMllamaSingleImage(policy, []string{"mllama"}, req); err != nil {
		t.Fatalf("historical expanded video should not block text follow-up: %v", err)
	}
}
