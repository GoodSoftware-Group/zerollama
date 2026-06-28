package glmocr

import (
	"errors"
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
	patchDim := m.VisionModel.numChannels * m.VisionModel.temporalPatchSize * m.VisionModel.patchSize * m.VisionModel.patchSize
	numPatches := grid.Temporal * grid.Height * grid.Width
	want := patchDim * numPatches
	if len(pixelValues) != want {
		return nil, fmt.Errorf("processor_output pixel_values length %d != %d", len(pixelValues), want)
	}

	pv := ctx.Input().FromFloats(pixelValues, patchDim, numPatches)
	visionOutputs := m.VisionModel.Forward(ctx, pv, grid)
	if m.VisionDownsample == nil || m.VisionDownsample.Weight == nil {
		return nil, errors.New("glmocr: missing vision downsample weights")
	}
	visionOutputs = m.VisionDownsample.Forward(ctx, visionOutputs, grid, m.VisionModel.VisionModelOptions)
	if m.PatchMerger == nil {
		return nil, errors.New("glmocr: missing patch merger weights")
	}
	visionOutputs = m.PatchMerger.Forward(ctx, visionOutputs, m.VisionModel.VisionModelOptions)
	return []input.Multimodal{{Tensor: visionOutputs, Data: grid}}, nil
}
