package modality

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestPreflightVideoVisionBudget_preexpandedSpans(t *testing.T) {
	policy := VideoSamplingPolicy{MaxFrames: 8, TokensPerImage: 768}
	req := &api.ChatRequest{
		Messages: []api.Message{{
			Role:       "user",
			Images:     make([]api.ImageData, 5), // 1 still + 4 video frames
			VideoSpans: []api.VideoSpan{{FrameCount: 4}},
		}},
	}
	// 5 * 768 = 3840
	if err := PreflightVideoVisionBudget(policy, 4096, req); err != nil {
		t.Fatalf("expected pass under 4096: %v", err)
	}
	if err := PreflightVideoVisionBudget(policy, 3000, req); err == nil {
		t.Fatal("expected exceed num_ctx for 5 frames @ 768")
	}
}

func TestPreflightVideoVisionBudget_rawVideosUnchanged(t *testing.T) {
	policy := VideoSamplingPolicy{MaxFrames: 4, TokensPerImage: 100}
	req := &api.ChatRequest{
		Messages: []api.Message{{
			Images: []api.ImageData{{1}},
			Videos: []api.VideoData{{9}},
		}},
	}
	// 1 still + 4 max frames = 5 * 100 = 500
	if err := PreflightVideoVisionBudget(policy, 400, req); err == nil {
		t.Fatal("expected exceed")
	}
	if err := PreflightVideoVisionBudget(policy, 600, req); err != nil {
		t.Fatalf("expected pass: %v", err)
	}
}

func TestPreflightVideoVisionBudget_ignoresHistoricalPreexpanded(t *testing.T) {
	policy := VideoSamplingPolicy{MaxFrames: 8, TokensPerImage: 768}
	heavyHistory := make([]api.ImageData, 8)
	req := &api.ChatRequest{
		Messages: []api.Message{
			{
				Role:       "user",
				Images:     heavyHistory,
				VideoSpans: []api.VideoSpan{{FrameCount: 8}},
			},
			{Role: "assistant", Content: "ok"},
			{Role: "user", Content: "follow up"},
		},
	}
	// Only latest user turn has no multimodal payload — should pass.
	if err := PreflightVideoVisionBudget(policy, 1024, req); err != nil {
		t.Fatalf("historical spans should not inflate budget: %v", err)
	}
}

func TestPreflightVideoVisionBudget_countsLatestPreexpanded(t *testing.T) {
	policy := VideoSamplingPolicy{MaxFrames: 8, TokensPerImage: 100}
	req := &api.ChatRequest{
		Messages: []api.Message{
			{Role: "user", Content: "earlier"},
			{
				Role:       "user",
				Images:     make([]api.ImageData, 5),
				VideoSpans: []api.VideoSpan{{FrameCount: 5}},
			},
		},
	}
	if err := PreflightVideoVisionBudget(policy, 400, req); err == nil {
		t.Fatal("expected exceed from latest user spans only")
	}
}
