// Strip multimodal payload covered by prefix cache (vLLM #52041 analog).
//
// WHY: When num_computed_tokens covers an image/video span, downstream workers
// do not need raw bytes or embed tensors for that item — metadata suffices.
package modality

import (
	"github.com/ollama/ollama/llm"
)

// MediaSpan is a half-open token range [Start, End) for one multimodal item.
type MediaSpan struct {
	Start int
	End   int
}

// StripCoveredImageData nils ImageData.Data for items fully inside the computed
// prefix. Spans with End <= numComputed are stripped; items extending past the
// prefix keep their payload (vLLM strip_covered_mm_data semantics).
func StripCoveredImageData(images []llm.ImageData, spans []MediaSpan, numComputed int) []llm.ImageData {
	if len(images) == 0 || numComputed <= 0 || len(spans) == 0 {
		return images
	}
	if len(spans) != len(images) {
		return images
	}
	out := make([]llm.ImageData, len(images))
	copy(out, images)
	for i, span := range spans {
		if span.End <= numComputed {
			out[i].Data = nil
		}
	}
	return out
}
