// Preflight and wire helpers for SGLang processor_output (HF pixel_values + grid).
//
// WHY separate from precomputed: processor_output still runs vision tower on server;
// precomputed skips encode entirely. Mixing them on one message would make cache
// keys and runner inject paths ambiguous — SGLang rejects the same combination.
package modality

import (
	"fmt"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

// PreflightPreprocessedInputs validates SGLang precomputed_embedding and processor_output rules.
func PreflightPreprocessedInputs(req *api.ChatRequest) error {
	if err := PreflightPrecomputedEmbeddings(req); err != nil {
		return err
	}
	return PreflightProcessorOutputs(req)
}

// PreflightProcessorOutputs validates SGLang processor_output payloads on /api/chat.
func PreflightProcessorOutputs(req *api.ChatRequest) error {
	if req == nil {
		return nil
	}
	for i, msg := range req.Messages {
		if len(msg.ProcessorOutputs) == 0 {
			continue
		}
		if len(msg.ProcessorOutputs) != 1 {
			return fmt.Errorf("message[%d]: processor_output accepts exactly one item, got %d", i, len(msg.ProcessorOutputs))
		}
		if len(msg.Images) > 0 || len(msg.PrecomputedEmbeddings) > 0 {
			return fmt.Errorf("message[%d]: cannot mix processor_output with raw images or precomputed_embedding", i)
		}
		if len(msg.PaddedInputIDs) == 0 {
			return fmt.Errorf("message[%d]: processor_output requires padded_input_ids on the same message", i)
		}
		po := msg.ProcessorOutputs[0]
		if po.Format != "" && po.Format != api.ProcessorOutputFormat {
			return fmt.Errorf("message[%d]: unsupported processor format %q", i, po.Format)
		}
		if len(po.PixelValues) == 0 {
			return fmt.Errorf("message[%d]: processor_output requires pixel_values", i)
		}
		if len(po.GridTHWForProcessor()) != 3 {
			return fmt.Errorf("message[%d]: processor_output requires image_grid_thw [T,H,W]", i)
		}
	}
	return nil
}

// AppendProcessorOutputsToLLM appends llm.ImageData entries for processor_output on msg.
func AppendProcessorOutputsToLLM(msg api.Message, images []llm.ImageData) []llm.ImageData {
	for _, po := range msg.ProcessorOutputs {
		pv := append([]float32(nil), po.PixelValues...)
		img := llm.ImageData{
			ID:                   len(images),
			ProcessorPixelValues: pv,
		}
		if thw := po.GridTHWForProcessor(); len(thw) == 3 {
			img.GridTHW = append([]int(nil), thw...)
		}
		images = append(images, img)
	}
	return images
}
