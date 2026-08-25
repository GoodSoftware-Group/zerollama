package llm

import "context"

// visionSpanHints carries family-specific slot ids for padded-inject span maps.
// Zero values fall back to llama-server defaults (same as placeholder tokenize).
type visionSpanHints struct {
	Gemma4      Gemma4SoftTokens
	Gemma4Media Gemma4PaddedMediaSchedule
	Lfm2        Lfm2VisionTokens
	Mistral3    Mistral3VisionTokens
	Deepseek    DeepseekOcrVisionTokens
	Gemma3Slot  int
	MllamaSlot  int
}

func defaultVisionSpanHints() visionSpanHints {
	return visionSpanHints{
		Lfm2:       Lfm2VisionTokens{Image: 396},
		Mistral3:   Mistral3VisionTokens{Img: 10, Break: 12, End: 13},
		Deepseek:   DeepseekOcrVisionTokens{Image: 128815},
		Gemma3Slot: gemma3ImageSlotTokenDefault,
		MllamaSlot: mllamaImageSlotTokenDefault,
	}
}

func mergeVisionSpanHints(h visionSpanHints) visionSpanHints {
	d := defaultVisionSpanHints()
	if h.Gemma4.Image != 0 || h.Gemma4.Video != 0 || h.Gemma4.Audio != 0 {
		d.Gemma4 = h.Gemma4
	}
	d.Gemma4Media = h.Gemma4Media
	if h.Lfm2.Image != 0 || h.Lfm2.UseBlock {
		d.Lfm2 = h.Lfm2
	}
	if h.Mistral3.Img != 0 {
		d.Mistral3 = h.Mistral3
	}
	if h.Deepseek.Image != 0 {
		d.Deepseek = h.Deepseek
	}
	if h.Gemma3Slot != 0 {
		d.Gemma3Slot = h.Gemma3Slot
	}
	if h.MllamaSlot != 0 {
		d.MllamaSlot = h.MllamaSlot
	}
	return d
}

func visionMediaSpansFromPromptTokens(tokens []int, paddedLayoutConsume string, hints visionSpanHints) []MediaSpan {
	h := mergeVisionSpanHints(hints)
	switch paddedLayoutConsume {
	case "", PaddedLayoutConsumeQwen3VLHFRunner:
		return qwen3VLVisionMediaSpans(tokens)
	case PaddedLayoutConsumeGemma4ImgRunner:
		return gemma4VisionMediaSpans(tokens, h.Gemma4, h.Gemma4Media)
	case PaddedLayoutConsumeGemma3ImgRunner:
		return slotTokenSpans(tokens, h.Gemma3Slot)
	case PaddedLayoutConsumeMllamaImgRunner:
		return slotTokenSpans(tokens, h.MllamaSlot)
	case PaddedLayoutConsumeLlama4ImgRunner:
		return llama4VisionMediaSpans(tokens)
	case PaddedLayoutConsumeLfm2ImgRunner, PaddedLayoutConsumeGlmocrImgRunner:
		return lfm2VisionMediaSpans(tokens, h.Lfm2)
	case PaddedLayoutConsumeMistral3ImgRunner:
		return mistral3VisionMediaSpans(tokens, h.Mistral3)
	case PaddedLayoutConsumeDeepseekOcrImgRunner:
		return imageRunSpans(tokens, h.Deepseek.Image)
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

func slotTokenSpans(tokens []int, slot int) []MediaSpan {
	if slot == 0 {
		return nil
	}
	var spans []MediaSpan
	for i, t := range tokens {
		if t == slot {
			spans = append(spans, MediaSpan{Start: i, End: i + 1})
		}
	}
	return spans
}

func gemma4VisionMediaSpans(tokens []int, slots Gemma4SoftTokens, schedule Gemma4PaddedMediaSchedule) []MediaSpan {
	if slots.Image == 0 && slots.Video == 0 && slots.Audio == 0 {
		return nil
	}
	var spans []MediaSpan
	videoIdx := 0
	for i, t := range tokens {
		switch t {
		case slots.Image, slots.Audio:
			if t != 0 {
				spans = append(spans, MediaSpan{Start: i, End: i + 1})
			}
		case slots.Video:
			if slots.Video == 0 {
				continue
			}
			n := 1
			if videoIdx < len(schedule.VideoFrameCounts) {
				n = schedule.VideoFrameCounts[videoIdx]
			}
			if n <= 0 {
				n = 1
			}
			for range n {
				spans = append(spans, MediaSpan{Start: i, End: i + 1})
			}
			videoIdx++
		}
	}
	return spans
}

func llama4VisionMediaSpans(tokens []int) []MediaSpan {
	var spans []MediaSpan
	for i := 0; i < len(tokens); i++ {
		if !isLlama4ImageBlockStart(tokens, i) {
			continue
		}
		end := llama4ImageBlockEndIndex(tokens, i)
		spans = append(spans, MediaSpan{Start: i, End: end + 1})
		i = end
	}
	return spans
}

func lfm2VisionMediaSpans(tokens []int, slots Lfm2VisionTokens) []MediaSpan {
	if slots.UseBlock {
		var spans []MediaSpan
		for i := 0; i < len(tokens); i++ {
			if !isLfm2ImageBlockStart(tokens, i, slots) {
				continue
			}
			end := lfm2ImageBlockEndIndex(tokens, i, slots.End)
			spans = append(spans, MediaSpan{Start: i, End: end + 1})
			i = end
		}
		return spans
	}
	return imageRunSpans(tokens, slots.Image)
}

func mistral3VisionMediaSpans(tokens []int, slots Mistral3VisionTokens) []MediaSpan {
	var spans []MediaSpan
	for i := 0; i < len(tokens); i++ {
		if !isMistral3ImageInjectStart(tokens, i, slots) {
			continue
		}
		end := mistral3ImageBlockEndIndex(tokens, i, slots)
		spans = append(spans, MediaSpan{Start: i, End: end + 1})
		i = end
	}
	return spans
}

func imageRunSpans(tokens []int, imageTok int) []MediaSpan {
	if imageTok == 0 {
		return nil
	}
	var spans []MediaSpan
	for i := 0; i < len(tokens); i++ {
		if tokens[i] != imageTok || !isFirstImageTokenInRun(tokens, i, imageTok) {
			continue
		}
		end := skipImageTokenRun(tokens, i, imageTok) + 1
		spans = append(spans, MediaSpan{Start: i, End: end})
		i = end - 1
	}
	return spans
}

func stripCoveredCompletionMedia(req *CompletionRequest, numComputed int, hints visionSpanHints) {
	if req == nil || numComputed <= 0 || len(req.PromptTokens) == 0 {
		return
	}
	if hints.Gemma4Media.StillImageCount == 0 && hints.Gemma4Media.AudioClipCount == 0 && len(hints.Gemma4Media.VideoFrameCounts) == 0 {
		hints.Gemma4Media = req.Gemma4PaddedMedia
	}
	spans := visionMediaSpansFromPromptTokens(req.PromptTokens, req.PaddedLayoutConsume, hints)
	if len(spans) == 0 {
		return
	}
	req.Images = StripCoveredImageData(req.Images, spans, numComputed)
	req.Media = StripCoveredMediaData(req.Media, spans, numComputed)
}

func (s *llamaServerRunner) visionSpanHints(ctx context.Context, req CompletionRequest) visionSpanHints {
	h := visionSpanHints{Gemma4Media: req.Gemma4PaddedMedia}
	if s == nil {
		return h
	}
	switch req.PaddedLayoutConsume {
	case PaddedLayoutConsumeGemma4ImgRunner:
		if slots, err := s.gemma4SoftTokens(ctx); err == nil {
			h.Gemma4 = slots
		}
	case PaddedLayoutConsumeGemma3ImgRunner:
		h.Gemma3Slot = gemma3ImageSlotTokenDefault
		if id, err := s.placeholderToken(ctx, "<start_of_image>"); err == nil && id != 0 {
			h.Gemma3Slot = id
		}
	case PaddedLayoutConsumeMllamaImgRunner:
		h.MllamaSlot = mllamaImageSlotTokenDefault
		if id, err := s.placeholderToken(ctx, "<|image|>"); err == nil && id != 0 {
			h.MllamaSlot = id
		}
	case PaddedLayoutConsumeLfm2ImgRunner:
		if slots, err := s.lfm2VisionTokens(ctx); err == nil {
			h.Lfm2 = slots
		}
	case PaddedLayoutConsumeGlmocrImgRunner:
		if slots, err := s.glmocrVisionTokens(ctx); err == nil {
			h.Lfm2 = slots
		}
	case PaddedLayoutConsumeMistral3ImgRunner:
		if slots, err := s.mistral3VisionTokens(ctx); err == nil {
			h.Mistral3 = slots
		}
	case PaddedLayoutConsumeDeepseekOcrImgRunner:
		if slots, err := s.deepseekOcrVisionTokens(ctx); err == nil {
			h.Deepseek = slots
		}
	}
	return h
}
