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
	stripCoveredCompletionMedia(req, 5, visionSpanHints{})
	if req.Images[0].Data != nil {
		t.Fatal("expected first image stripped")
	}
	if req.Images[1].Data == nil {
		t.Fatal("second image has no span — keep payload")
	}
	if req.Media[0].Data != nil {
		t.Fatal("expected first media stripped")
	}
}

func TestGemma4VisionMediaSpans_videoFrames(t *testing.T) {
	slots := Gemma4SoftTokens{Image: 42, Video: 43, Audio: 44}
	tokens := []int{1, 42, 2, 43, 3}
	spans := gemma4VisionMediaSpans(tokens, slots, Gemma4PaddedMediaSchedule{VideoFrameCounts: []int{3}})
	if len(spans) != 4 {
		t.Fatalf("spans=%v want 1 image + 3 video frames", spans)
	}
	if spans[0].Start != 1 || spans[1].Start != 3 {
		t.Fatalf("span starts=%v", spans)
	}
}

func TestLlama4VisionMediaSpans(t *testing.T) {
	tokens := []int{1, llama4ImageBoundary, llama4PatchToken, llama4ImageBoundary, 9}
	spans := llama4VisionMediaSpans(tokens)
	if len(spans) != 1 || spans[0].Start != 1 || spans[0].End != 4 {
		t.Fatalf("spans=%v", spans)
	}
}

func TestImageRunSpans_deepseek(t *testing.T) {
	const tok = 128815
	tokens := []int{1, tok, tok, tok, 2}
	spans := imageRunSpans(tokens, tok)
	if len(spans) != 1 || spans[0].Start != 1 || spans[0].End != 4 {
		t.Fatalf("spans=%v", spans)
	}
}

func TestStripCoveredCompletionMedia_llama4Partial(t *testing.T) {
	req := &CompletionRequest{
		PaddedLayoutConsume: PaddedLayoutConsumeLlama4ImgRunner,
		PromptTokens: []int{
			1, llama4ImageBoundary, llama4PatchToken, llama4ImageBoundary, 9,
			llama4ImageBoundary, llama4PatchToken, llama4ImageBoundary,
		},
		Images: []ImageData{
			{ID: 0, Data: []byte("a")},
			{ID: 1, Data: []byte("b")},
		},
	}
	stripCoveredCompletionMedia(req, 5, visionSpanHints{})
	if req.Images[0].Data != nil {
		t.Fatal("first llama4 block should be stripped")
	}
	if req.Images[1].Data == nil {
		t.Fatal("second llama4 block should keep payload")
	}
}

func TestLfm2VisionMediaSpans_imageRun(t *testing.T) {
	slots := Lfm2VisionTokens{Image: 396}
	tokens := []int{10, 396, 396, 11}
	spans := lfm2VisionMediaSpans(tokens, slots)
	if len(spans) != 1 || spans[0].End != 3 {
		t.Fatalf("spans=%v", spans)
	}
}
