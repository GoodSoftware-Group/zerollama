package modality

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestVideoExpandCache_hitSkipsDecodeHook(t *testing.T) {
	resetVideoExpandCache()
	resetSessionVideoExpandCache()
	policy := VideoSamplingPolicy{Mode: "fps", FPS: 1, MaxFrames: 8, MaxVideosPerMessage: 4, MaxImagesAfterExpand: 64, MaxBytes: 1 << 20}
	video := []byte{0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70} // fake mp4 header

	var calls atomic.Int32
	orig := ExternalVideoDecodeHook
	defer func() { ExternalVideoDecodeHook = orig }()
	ExternalVideoDecodeHook = func(ctx context.Context, policy VideoSamplingPolicy, data []byte) ([]api.ImageData, error) {
		calls.Add(1)
		return []api.ImageData{{0x89, 0x50}}, nil
	}

	req := &api.ChatRequest{
		Messages: []api.Message{{Videos: []api.VideoData{video}}},
	}
	if err := ExpandVideosInChatRequest(context.Background(), policy, req); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 1 {
		t.Fatalf("first expand calls=%d, want 1", calls.Load())
	}

	req2 := &api.ChatRequest{
		Messages: []api.Message{{Videos: []api.VideoData{video}}},
	}
	if err := ExpandVideosInChatRequest(context.Background(), policy, req2); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 1 {
		t.Fatalf("cached expand calls=%d, want still 1", calls.Load())
	}
	if len(req2.Messages[0].Images) != 1 {
		t.Fatalf("cached frames=%d, want 1", len(req2.Messages[0].Images))
	}
}

func TestVideoExpandCache_policyChangeMisses(t *testing.T) {
	resetVideoExpandCache()
	resetSessionVideoExpandCache()
	video := []byte{0x01, 0x02, 0x03}

	var calls atomic.Int32
	orig := ExternalVideoDecodeHook
	defer func() { ExternalVideoDecodeHook = orig }()
	ExternalVideoDecodeHook = func(ctx context.Context, policy VideoSamplingPolicy, data []byte) ([]api.ImageData, error) {
		calls.Add(1)
		return []api.ImageData{{0xaa}}, nil
	}

	p1 := VideoSamplingPolicy{Mode: "fps", FPS: 1, MaxFrames: 4, MaxVideosPerMessage: 4, MaxImagesAfterExpand: 64, MaxBytes: 1 << 20}
	p2 := VideoSamplingPolicy{Mode: "fps", FPS: 2, MaxFrames: 4, MaxVideosPerMessage: 4, MaxImagesAfterExpand: 64, MaxBytes: 1 << 20}
	for _, policy := range []VideoSamplingPolicy{p1, p2} {
		req := &api.ChatRequest{Messages: []api.Message{{Videos: []api.VideoData{video}}}}
		if err := ExpandVideosInChatRequest(context.Background(), policy, req); err != nil {
			t.Fatal(err)
		}
	}
	if calls.Load() != 2 {
		t.Fatalf("policy change calls=%d, want 2", calls.Load())
	}
}

func TestLookupVideoExpandCache_emptyDataMiss(t *testing.T) {
	resetVideoExpandCache()
	if _, ok := lookupVideoExpandCache(VideoSamplingPolicy{}, nil); ok {
		t.Fatal("expected miss for nil data")
	}
}

func TestRememberVideoExpandCache_emptyFramesNoOp(t *testing.T) {
	resetVideoExpandCache()
	rememberVideoExpandCache(VideoSamplingPolicy{}, []byte{1}, nil, nil)
	if _, ok := lookupVideoExpandCache(VideoSamplingPolicy{}, []byte{1}); ok {
		t.Fatal("empty frames should not be cached")
	}
}

func TestExternalVideoDecodeHook_errorNotCached(t *testing.T) {
	resetVideoExpandCache()
	orig := ExternalVideoDecodeHook
	defer func() { ExternalVideoDecodeHook = orig }()
	ExternalVideoDecodeHook = func(ctx context.Context, policy VideoSamplingPolicy, data []byte) ([]api.ImageData, error) {
		return nil, errors.New("boom")
	}
	req := &api.ChatRequest{Messages: []api.Message{{Videos: []api.VideoData{{9}}}}}
	if err := ExpandVideosInChatRequest(context.Background(), VideoSamplingPolicy{MaxVideosPerMessage: 4, MaxImagesAfterExpand: 64, MaxBytes: 1 << 20}, req); err == nil {
		t.Fatal("expected error")
	}
	if _, ok := lookupVideoExpandCache(VideoSamplingPolicy{}, []byte{9}); ok {
		t.Fatal("failed decode should not populate cache")
	}
}
