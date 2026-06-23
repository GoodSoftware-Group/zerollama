package modality

import (
	"context"
	"sync/atomic"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestPolicyFingerprint_deterministic(t *testing.T) {
	p1 := VideoSamplingPolicy{Mode: "fps", FPS: 1.5, Stride: 0, MaxFrames: 8}
	p2 := VideoSamplingPolicy{Mode: "fps", FPS: 1.5, Stride: 0, MaxFrames: 8}
	if policyFingerprint(p1) != policyFingerprint(p2) {
		t.Fatal("same policy should fingerprint equally")
	}
	p3 := VideoSamplingPolicy{Mode: "fps", FPS: 2, Stride: 0, MaxFrames: 8}
	if policyFingerprint(p1) == policyFingerprint(p3) {
		t.Fatal("fps change should change fingerprint")
	}
}

func TestExpandVideosInChatRequest_multiClipSpansOrder(t *testing.T) {
	resetVideoExpandCache()
	resetSessionVideoExpandCache()

	var calls atomic.Int32
	orig := ExternalVideoDecodeHook
	defer func() { ExternalVideoDecodeHook = orig }()
	ExternalVideoDecodeHook = func(ctx context.Context, policy VideoSamplingPolicy, data []byte) ([]api.ImageData, error) {
		calls.Add(1)
		n := int(data[0])
		if n <= 0 {
			n = 1
		}
		out := make([]api.ImageData, n)
		for i := range out {
			out[i] = []byte{data[0], byte(i)}
		}
		return out, nil
	}

	policy := VideoSamplingPolicy{
		Mode:                 "fps",
		FPS:                  1,
		MaxFrames:            8,
		MaxVideosPerMessage:  4,
		MaxImagesAfterExpand: 64,
		MaxBytes:             1 << 20,
	}
	req := &api.ChatRequest{
		Messages: []api.Message{{
			Videos: []api.VideoData{{2}, {3}},
		}},
	}
	if err := ExpandVideosInChatRequest(context.Background(), policy, req); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 2 {
		t.Fatalf("decode calls=%d want 2", calls.Load())
	}
	if len(req.Messages[0].VideoSpans) != 2 {
		t.Fatalf("spans=%d want 2", len(req.Messages[0].VideoSpans))
	}
	if req.Messages[0].VideoSpans[0].FrameCount != 2 || req.Messages[0].VideoSpans[1].FrameCount != 3 {
		t.Fatalf("span counts=%v", req.Messages[0].VideoSpans)
	}
	if len(req.Messages[0].Images) != 5 {
		t.Fatalf("images=%d want 5", len(req.Messages[0].Images))
	}
}

func TestVideoExpandCacheKey_policyIsolates(t *testing.T) {
	resetVideoExpandCache()
	data := []byte{0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70}
	p1 := VideoSamplingPolicy{Mode: "fps", FPS: 1, MaxFrames: 4, MaxVideosPerMessage: 4, MaxImagesAfterExpand: 64, MaxBytes: 1 << 20}
	p2 := VideoSamplingPolicy{Mode: "fps", FPS: 2, MaxFrames: 4, MaxVideosPerMessage: 4, MaxImagesAfterExpand: 64, MaxBytes: 1 << 20}

	rememberVideoExpandCache(p1, data, []api.ImageData{{0xaa}}, []int{1, 4, 4})
	if _, ok := lookupVideoExpandCache(p2, data); ok {
		t.Fatal("different fps policy should not share cache entry")
	}
	if _, ok := lookupVideoExpandCache(p1, data); !ok {
		t.Fatal("expected hit for matching policy")
	}
}
