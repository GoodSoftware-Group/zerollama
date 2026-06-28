// Preflight and wire helpers for SGLang precomputed_embedding payloads.
//
// WHY preflight before ffmpeg: invalid row shapes or mixed modalities should 400
// before ViT work — same ordering as SGLang's limit_mm_data checks. Requires
// padded_input_ids on the same message because vision token positions are client-owned.
package modality

import (
	"fmt"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

// MessageHasPrecomputedEmbeddings reports whether any message carries SGLang precomputed rows.
func MessageHasPrecomputedEmbeddings(msgs []api.Message) bool {
	for _, msg := range msgs {
		if len(msg.PrecomputedEmbeddings) > 0 {
			return true
		}
	}
	return false
}

// PreflightPrecomputedEmbeddings validates SGLang precomputed_embedding rules before inference.
func PreflightPrecomputedEmbeddings(req *api.ChatRequest) error {
	if req == nil {
		return nil
	}
	for i, msg := range req.Messages {
		if len(msg.PrecomputedEmbeddings) == 0 {
			continue
		}
		if len(msg.ProcessorOutputs) > 0 {
			return fmt.Errorf("message[%d]: cannot mix precomputed_embedding and processor_output", i)
		}
		if len(msg.PrecomputedEmbeddings) != 1 {
			return fmt.Errorf("message[%d]: precomputed_embedding accepts exactly one item, got %d", i, len(msg.PrecomputedEmbeddings))
		}
		if len(msg.Images) > 0 {
			return fmt.Errorf("message[%d]: cannot mix raw images and precomputed_embedding", i)
		}
		if len(msg.PaddedInputIDs) == 0 {
			return fmt.Errorf("message[%d]: precomputed_embedding requires padded_input_ids on the same message", i)
		}
		pe := msg.PrecomputedEmbeddings[0]
		if pe.Format != "" && pe.Format != api.PrecomputedEmbeddingFormat {
			return fmt.Errorf("message[%d]: unsupported precomputed format %q", i, pe.Format)
		}
		if err := validatePrecomputedFeature(pe.Feature); err != nil {
			return fmt.Errorf("message[%d]: %w", i, err)
		}
	}
	return nil
}

func validatePrecomputedFeature(rows [][]float32) error {
	if len(rows) == 0 {
		return fmt.Errorf("precomputed feature must have at least one row")
	}
	width := len(rows[0])
	if width == 0 {
		return fmt.Errorf("precomputed feature rows must be non-empty")
	}
	for i, row := range rows {
		if len(row) != width {
			return fmt.Errorf("precomputed feature row %d width %d != first row %d", i, len(row), width)
		}
	}
	return nil
}

// AppendPrecomputedImagesToLLM appends llm.ImageData entries for precomputed embeddings on msg.
func AppendPrecomputedImagesToLLM(msg api.Message, images []llm.ImageData) []llm.ImageData {
	for _, pe := range msg.PrecomputedEmbeddings {
		rows := make([][]float32, len(pe.Feature))
		for i, row := range pe.Feature {
			rows[i] = append([]float32(nil), row...)
		}
		images = append(images, llm.ImageData{
			ID:                 len(images),
			PrecomputedFeature: rows,
		})
		if len(pe.GridTHW) == 3 {
			images[len(images)-1].GridTHW = append([]int(nil), pe.GridTHW...)
		}
	}
	return images
}
