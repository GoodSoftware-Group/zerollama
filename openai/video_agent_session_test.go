package openai

import (
	"context"
	"encoding/base64"
	"sync/atomic"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/server/modality"
	"github.com/ollama/ollama/types/model"
)

// TestOpenAIVideoAgentSessionCache_secondTurn proves /v1-shaped requests pin session
// expansion cache via prompt_cache_key (SGLang agent loop over OpenAI video_url).
func TestOpenAIVideoAgentSessionCache_secondTurn(t *testing.T) {
	modality.ResetExpandCachesForTest()

	var calls atomic.Int32
	orig := modality.ExternalVideoDecodeHook
	defer func() { modality.ExternalVideoDecodeHook = orig }()
	modality.ExternalVideoDecodeHook = func(ctx context.Context, policy modality.VideoSamplingPolicy, data []byte) ([]api.ImageData, error) {
		calls.Add(1)
		return []api.ImageData{{0x89, 0x50}, {0x89, 0x51}}, nil
	}

	videoBytes := []byte{0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70}
	vidURL := "data:video/mp4;base64," + base64.StdEncoding.EncodeToString(videoBytes)
	cacheKey := "openai-video-agent-1"
	policy := modality.ResolveVideoPolicy(model.ConfigV2{})

	turn1 := ChatCompletionRequest{
		Model:          "vl",
		PromptCacheKey: &cacheKey,
		Messages: []Message{{
			Role: "user",
			Content: []any{
				map[string]any{"type": "text", "text": "describe"},
				map[string]any{"type": "video_url", "video_url": map[string]any{"url": vidURL}},
			},
		}},
	}
	chat1, err := FromChatRequest(turn1)
	if err != nil {
		t.Fatal(err)
	}
	if chat1.Options["prompt_cache_key"] != cacheKey {
		t.Fatalf("prompt_cache_key=%v", chat1.Options["prompt_cache_key"])
	}
	if err := modality.ExpandVideosInChatRequest(context.Background(), policy, chat1); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 1 {
		t.Fatalf("turn1 decode calls=%d want 1", calls.Load())
	}

	turn2 := ChatCompletionRequest{
		Model:          "vl",
		PromptCacheKey: &cacheKey,
		Messages: []Message{
			{Role: "user", Content: "describe"},
			{Role: "assistant", Content: "pattern"},
			{
				Role: "user",
				Content: []any{
					map[string]any{"type": "text", "text": "again"},
					map[string]any{"type": "video_url", "video_url": map[string]any{"url": vidURL}},
				},
			},
		},
	}
	chat2, err := FromChatRequest(turn2)
	if err != nil {
		t.Fatal(err)
	}
	if err := modality.ExpandVideosInChatRequest(context.Background(), policy, chat2); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 1 {
		t.Fatalf("turn2 decode calls=%d want 1 (session cache)", calls.Load())
	}
	if len(chat2.Messages[2].Images) != 2 {
		t.Fatalf("turn2 images=%d want 2", len(chat2.Messages[2].Images))
	}
}
