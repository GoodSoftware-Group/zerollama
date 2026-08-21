// Strip multimodal payload covered by prefix cache (vLLM #52041 analog).
package modality

import (
	"github.com/ollama/ollama/llm"
)

// MediaSpan is a half-open token range [Start, End) for one multimodal item.
type MediaSpan = llm.MediaSpan

// StripCoveredImageData nils ImageData.Data for items fully inside the computed prefix.
func StripCoveredImageData(images []llm.ImageData, spans []MediaSpan, numComputed int) []llm.ImageData {
	return llm.StripCoveredImageData(images, spans, numComputed)
}
