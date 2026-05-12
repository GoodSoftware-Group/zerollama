package modality

import (
	"context"
	"errors"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestExpandVideosInChatRequest_usesPolicyAndSpans(t *testing.T) {
	t.Parallel()
	policy := ResolveVideoPolicy(model.ConfigV2{})
	orig := ExternalVideoDecodeHook
	defer func() { ExternalVideoDecodeHook = orig }()

	ExternalVideoDecodeHook = func(ctx context.Context, policy VideoSamplingPolicy, video []byte) ([]api.ImageData, error) {
		if len(video) == 0 {
			return nil, errors.New("empty")
		}
		return []api.ImageData{{0x89, 0x50}, {0x89, 0x50}}, nil
	}

	req := &api.ChatRequest{
		Messages: []api.Message{
			{Videos: []api.VideoData{{1}}},
		},
	}
	if err := ExpandVideosInChatRequest(context.Background(), policy, req); err != nil {
		t.Fatal(err)
	}
	if len(req.Messages[0].Images) != 2 || len(req.Messages[0].VideoSpans) != 1 || req.Messages[0].VideoSpans[0].FrameCount != 2 {
		t.Fatalf("unexpected expansion: images=%d spans=%v", len(req.Messages[0].Images), req.Messages[0].VideoSpans)
	}
}
