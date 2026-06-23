package modality

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestValidatePaddedInputIDs(t *testing.T) {
	t.Parallel()
	if err := validatePaddedInputIDs(nil); err != nil {
		t.Fatal(err)
	}
	if err := validatePaddedInputIDs([]int{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	if err := validatePaddedInputIDs([]int{-1}); err == nil {
		t.Fatal("expected negative id error")
	}
	long := make([]int, maxPaddedInputIDsLen+1)
	if err := validatePaddedInputIDs(long); err == nil {
		t.Fatal("expected length error")
	}
}

func TestPreflightVideoVisionBudget_paddedInputIDs(t *testing.T) {
	policy := VideoSamplingPolicy{TokensPerImage: 768}
	req := &api.ChatRequest{
		Messages: []api.Message{{
			Role:             "user",
			Images:           make([]api.ImageData, 4),
			VideoSpans:       []api.VideoSpan{{FrameCount: 4, GridTHW: []int{4, 24, 32}}},
			PaddedInputIDs:   make([]int, 900),
		}},
	}
	if err := PreflightVideoVisionBudget(policy, 800, req); err == nil {
		t.Fatal("expected exceed from padded_input_ids len")
	}
	if err := PreflightVideoVisionBudget(policy, 1000, req); err != nil {
		t.Fatalf("expected pass: %v", err)
	}
}

func TestEstimateMultimodalTokens_paddedInputIDsVideo(t *testing.T) {
	t.Parallel()
	policy := VideoSamplingPolicy{TokensPerImage: 768}
	req := &api.ChatRequest{
		Messages: []api.Message{{
			Role:           "user",
			Images:         make([]api.ImageData, 4),
			VideoSpans:     []api.VideoSpan{{FrameCount: 4}},
			PaddedInputIDs: make([]int, 500),
		}},
	}
	got := EstimateMultimodalTokens(policy, req)
	if got.VideoTokens != 500 {
		t.Fatalf("video_tokens=%d want 500 from padded_input_ids", got.VideoTokens)
	}
	if got.ImageTokens != 0 {
		t.Fatalf("image_tokens=%d want 0", got.ImageTokens)
	}
}

func TestExpandVideosInChatRequest_preprocessedPaddedInputIDs(t *testing.T) {
	req := &api.ChatRequest{
		Messages: []api.Message{{
			Images:         []api.ImageData{{1}, {2}},
			VideoSpans:     []api.VideoSpan{{FrameCount: 2}},
			PaddedInputIDs: []int{10, 20, 30},
		}},
	}
	if err := ExpandVideosInChatRequest(nil, VideoSamplingPolicy{MaxVideosPerMessage: 4, MaxImagesAfterExpand: 64, MaxBytes: 1 << 20}, req); err != nil {
		t.Fatal(err)
	}
}

func TestExpandVideosInChatRequest_rejectsInvalidPaddedInputIDs(t *testing.T) {
	req := &api.ChatRequest{
		Messages: []api.Message{{
			Images:         []api.ImageData{{1}},
			VideoSpans:     []api.VideoSpan{{FrameCount: 1}},
			PaddedInputIDs: []int{-1},
		}},
	}
	err := ExpandVideosInChatRequest(nil, VideoSamplingPolicy{MaxVideosPerMessage: 4, MaxImagesAfterExpand: 64, MaxBytes: 1 << 20}, req)
	if err == nil || !strings.Contains(err.Error(), "padded_input_ids") {
		t.Fatalf("err=%v", err)
	}
}

func TestChatRequestHasMultimodalPayload_paddedInputIDs(t *testing.T) {
	t.Parallel()
	req := &api.ChatRequest{
		Messages: []api.Message{{PaddedInputIDs: []int{1}}},
	}
	if !ChatRequestHasMultimodalPayload(req) {
		t.Fatal("expected multimodal from padded_input_ids")
	}
}

func TestLatestUserPaddedLayout_latestUserOnly(t *testing.T) {
	t.Parallel()
	req := &api.ChatRequest{
		Options: map[string]any{"prompt_cache_key": "thread-1"},
		Messages: []api.Message{
			{Role: "user", PaddedInputIDs: []int{1, 2, 3}},
			{Role: "assistant", Content: "ok"},
			{Role: "user", PaddedInputIDs: []int{10, 20}, VideoSpans: []api.VideoSpan{{FrameCount: 2}}},
		},
	}
	stub, ok := LatestUserPaddedLayout(req)
	if !ok {
		t.Fatal("expected layout on latest user")
	}
	if stub.Len != 2 {
		t.Fatalf("len=%d want 2", stub.Len)
	}
	if !stub.HasVideoSpans {
		t.Fatal("expected has_video_spans")
	}
	if stub.SessionKey != "thread-1" {
		t.Fatalf("session_key=%q", stub.SessionKey)
	}
}

func TestLatestUserPaddedLayout_miss(t *testing.T) {
	t.Parallel()
	if _, ok := LatestUserPaddedLayout(nil); ok {
		t.Fatal("nil req should miss")
	}
	if _, ok := LatestUserPaddedLayout(&api.ChatRequest{
		Messages: []api.Message{{Role: "user", Content: "hi"}},
	}); ok {
		t.Fatal("text-only should miss")
	}
}
