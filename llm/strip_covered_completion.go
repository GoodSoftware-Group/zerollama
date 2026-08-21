package llm

// visionMediaSpansFromPromptTokens maps pretokenized vision blocks to media indices
// in marker order (one span per vision block / media slot).
func visionMediaSpansFromPromptTokens(tokens []int, paddedLayoutConsume string) []MediaSpan {
	switch paddedLayoutConsume {
	case "", PaddedLayoutConsumeQwen3VLHFRunner:
		return qwen3VLVisionMediaSpans(tokens)
	default:
		return nil
	}
}

func qwen3VLVisionMediaSpans(tokens []int) []MediaSpan {
	ranges := qwen3VLVisionBlockRanges(tokens)
	if len(ranges) == 0 {
		return nil
	}
	spans := make([]MediaSpan, len(ranges))
	for i, r := range ranges {
		spans[i] = MediaSpan{Start: r[0], End: r[1] + 1}
	}
	return spans
}

func stripCoveredCompletionMedia(req *CompletionRequest, numComputed int) {
	if req == nil || numComputed <= 0 || len(req.PromptTokens) == 0 {
		return
	}
	spans := visionMediaSpansFromPromptTokens(req.PromptTokens, req.PaddedLayoutConsume)
	if len(spans) == 0 {
		return
	}
	if len(req.Images) >= len(spans) {
		req.Images = StripCoveredImageData(req.Images[:len(spans)], spans, numComputed)
	}
	if len(req.Media) >= len(spans) {
		req.Media = StripCoveredMediaData(req.Media[:len(spans)], spans, numComputed)
	}
}
