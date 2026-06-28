package qwen25vl

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
	patchDim := m.numChannels * m.temporalPatchSize * m.patchSize * m.patchSize
	numPatches := grid.Temporal * grid.Height * grid.Width
	want := patchDim * numPatches
	if len(pixelValues) != want {
		return nil, fmt.Errorf("processor_output pixel_values length %d != %d", len(pixelValues), want)
	}
	pv := ctx.Input().FromFloats(pixelValues, patchDim, numPatches)
	visionOutputs := m.VisionModel.Forward(ctx, pv, grid)
	return []input.Multimodal{{Tensor: visionOutputs, Data: grid}}, nil
}
