package llm

import "testing"

func TestSessionPrefixTracker_estimateExtendedPrompt(t *testing.T) {
	var tr sessionPrefixTracker
	tr.record("model.gguf", "agent:1", []int{1, 2, 3, 4, 5}, false)
	got := tr.estimate("model.gguf", "agent:1", []int{1, 2, 3, 4, 5, 6, 7}, false)
	if got != 4 {
		t.Fatalf("estimate=%d want 4 (len(prev)-1)", got)
	}
}

func TestSessionPrefixTracker_cacheReset(t *testing.T) {
	var tr sessionPrefixTracker
	tr.record("m", "k", []int{1, 2, 3}, false)
	tr.record("m", "k", nil, true)
	if tr.estimate("m", "k", []int{1, 2, 3, 4}, false) != 0 {
		t.Fatal("expected miss after cache_reset")
	}
}

func TestQwen3VLVisionMediaSpans(t *testing.T) {
	tokens := []int{10, qwenVLVisionStart, 151655, qwenVLVisionEnd, 20, qwenVLVisionStart, 151655, 151655, qwenVLVisionEnd}
	spans := qwen3VLVisionMediaSpans(tokens)
	if len(spans) != 2 {
		t.Fatalf("spans=%v", spans)
	}
	if spans[0].End != 4 || spans[1].Start != 5 {
		t.Fatalf("unexpected span bounds: %+v", spans)
	}
}

func TestStripCoveredCompletionMedia(t *testing.T) {
	req := &CompletionRequest{
		PromptTokens: []int{10, qwenVLVisionStart, 151655, qwenVLVisionEnd, 99},
		Images: []ImageData{
			{ID: 0, Data: []byte("img0")},
			{ID: 1, Data: []byte("img1")},
		},
		Media: []MediaData{
			NewMediaData(0, []byte("img0")),
			NewMediaData(1, []byte("img1")),
		},
	}
	// Only one qwen vision block in tokens — strip first image when fully covered.
	req.Images = req.Images[:1]
	req.Media = req.Media[:1]
	stripCoveredCompletionMedia(req, 5)
	if req.Images[0].Data != nil {
		t.Fatal("expected covered image stripped")
	}
	if req.Media[0].Data != nil {
		t.Fatal("expected covered media stripped")
	}
}
