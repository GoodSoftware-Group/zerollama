package llm

// MediaSpan is a half-open token range [Start, End) for one multimodal item.
type MediaSpan struct {
	Start int
	End   int
}

// StripCoveredImageData nils ImageData.Data for items fully inside the computed
// prefix (vLLM strip_covered_mm_data semantics).
func StripCoveredImageData(images []ImageData, spans []MediaSpan, numComputed int) []ImageData {
	n := min(len(images), len(spans))
	if n == 0 || numComputed <= 0 {
		return images
	}
	out := make([]ImageData, len(images))
	copy(out, images)
	for i := 0; i < n; i++ {
		if spans[i].End <= numComputed {
			out[i].Data = nil
		}
	}
	return out
}

// StripCoveredMediaData nils MediaData.Data for covered spans.
func StripCoveredMediaData(media []MediaData, spans []MediaSpan, numComputed int) []MediaData {
	n := min(len(media), len(spans))
	if n == 0 || numComputed <= 0 {
		return media
	}
	out := make([]MediaData, len(media))
	copy(out, media)
	for i := 0; i < n; i++ {
		if spans[i].End <= numComputed {
			out[i].Data = nil
		}
	}
	return out
}
