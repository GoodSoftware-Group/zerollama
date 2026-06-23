package modality

import (
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestEstimateMultimodalTokens_stillsAndVideoFrames(t *testing.T) {
	t.Parallel()
	policy := VideoSamplingPolicy{TokensPerImage: 100}
	req := &api.ChatRequest{
		Messages: []api.Message{
			{
				Images:     make([]api.ImageData, 5), // 2 still + 3 video frames
				VideoSpans: []api.VideoSpan{{FrameCount: 3}},
			},
		},
	}
	got := EstimateMultimodalTokens(policy, req)
	if got.ImageTokens != 200 || got.VideoTokens != 300 {
		t.Fatalf("got image=%d video=%d, want 200/300", got.ImageTokens, got.VideoTokens)
	}
}

func TestEstimateMultimodalTokens_defaultTokensPerImage(t *testing.T) {
	t.Parallel()
	policy := ResolveVideoPolicy(model.ConfigV2{})
	req := &api.ChatRequest{
		Messages: []api.Message{{Images: []api.ImageData{{1}}}},
	}
	got := EstimateMultimodalTokens(policy, req)
	if got.ImageTokens != 768 {
		t.Fatalf("got %d, want default 768", got.ImageTokens)
	}
}

func TestEstimateMultimodalTokens_audioClips(t *testing.T) {
	t.Parallel()
	policy := VideoSamplingPolicy{TokensPerImage: 100}
	req := &api.ChatRequest{
		Messages: []api.Message{{AudioClips: []api.AudioData{{1}, {2}}}},
	}
	got := EstimateMultimodalTokens(policy, req)
	if got.AudioTokens != 200 {
		t.Fatalf("got %d, want 200", got.AudioTokens)
	}
}

func TestEstimateMultimodalTokens_agentHistoryNotDoubledCounted(t *testing.T) {
	t.Parallel()
	// Agent echoes turn-1 video_spans in history; only the latest user's spans should count.
	policy := VideoSamplingPolicy{TokensPerImage: 100}
	req := &api.ChatRequest{
		Messages: []api.Message{
			{
				Role:       "user",
				Images:     make([]api.ImageData, 3), // 3 frames from turn-1 video
				VideoSpans: []api.VideoSpan{{FrameCount: 3}},
			},
			{Role: "assistant", Content: "ok"},
			{
				Role:       "user",
				Images:     make([]api.ImageData, 2), // 2 frames from turn-2 video
				VideoSpans: []api.VideoSpan{{FrameCount: 2}},
			},
		},
	}
	got := EstimateMultimodalTokens(policy, req)
	// Only latest user message: 2 video frames * 100 = 200; historical turn-1 spans ignored.
	if got.VideoTokens != 200 {
		t.Fatalf("video_tokens=%d want 200 (latest user only)", got.VideoTokens)
	}
	if got.ImageTokens != 0 {
		t.Fatalf("image_tokens=%d want 0 (historical stills/spans ignored)", got.ImageTokens)
	}
}

func TestEstimateMultimodalTokens_ignoresHistoricalStills(t *testing.T) {
	t.Parallel()
	policy := VideoSamplingPolicy{TokensPerImage: 100}
	req := &api.ChatRequest{
		Messages: []api.Message{
			{Role: "user", Images: []api.ImageData{{1}, {2}}},
			{Role: "assistant", Content: "ok"},
			{Role: "user", Content: "follow up"},
		},
	}
	got := EstimateMultimodalTokens(policy, req)
	if got.HasValues() {
		t.Fatalf("expected zero for text-only latest turn, got %+v", got)
	}
}
