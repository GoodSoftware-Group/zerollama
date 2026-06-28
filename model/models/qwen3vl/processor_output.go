package qwen3vl

import (
	"fmt"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

// MultimodalFromProcessorOutput runs the vision tower on HF processor pixel_values.
func (m *Model) MultimodalFromProcessorOutput(ctx ml.Context, pixelValues []float32, gridTHW []int) ([]input.Multimodal, error) {
	grid, err := gridFromPrecomputedTHW(gridTHW)
	if err != nil {
		return nil, err
	}
	patchDim := m.ImageProcessor.numChannels * m.ImageProcessor.temporalPatchSize *
		m.ImageProcessor.patchSize * m.ImageProcessor.patchSize
	numPatches := grid.Temporal * grid.Height * grid.Width
	want := patchDim * numPatches
	if len(pixelValues) != want {
		return nil, fmt.Errorf("processor_output pixel_values length %d != %d (patch_dim=%d num_patches=%d)", len(pixelValues), want, patchDim, numPatches)
	}
	pv := ctx.Input().FromFloats(pixelValues, patchDim, numPatches)
	visionOutputs, deepstackVisualEmbeds := m.VisionModel.Forward(ctx, pv, grid)
	mm := []input.Multimodal{{Tensor: visionOutputs, Data: grid}}
	for i := range deepstackVisualEmbeds {
		mm = append(mm, input.Multimodal{Tensor: deepstackVisualEmbeds[i]})
	}
	return mm, nil
}
