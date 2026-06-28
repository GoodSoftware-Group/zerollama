package gemma4

import (
	"fmt"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

// MultimodalFromProcessorOutput runs the vision tower on HF processor pixel_values.
// image_grid_thw is interpreted as [1, pixel_height, pixel_width] (channel-first pixels).
func (m *Model) MultimodalFromProcessorOutput(ctx ml.Context, pixelValues []float32, gridTHW []int) ([]input.Multimodal, error) {
	imgW, imgH, err := gemma4PixelSizeFromGrid(gridTHW, m.ImageProcessor.numChannels, len(pixelValues))
	if err != nil {
		return nil, err
	}

	patchSize := m.ImageProcessor.patchSize
	if imgW%patchSize != 0 || imgH%patchSize != 0 {
		return nil, fmt.Errorf("processor_output pixel size %dx%d must be divisible by patch_size %d", imgW, imgH, patchSize)
	}

	pixelTensor := ctx.Input().FromFloats(pixelValues, imgW, imgH, m.ImageProcessor.numChannels)
	numPatchesX := imgW / patchSize
	numPatchesY := imgH / patchSize

	visionOutputs := m.VisionModel.Forward(ctx, pixelTensor, numPatchesX, numPatchesY)
	visionOutputs = visionPoolAndProject(ctx, visionOutputs, numPatchesX, numPatchesY, m.VisionModel.VisionModelOptions, m.MultiModalProjector, m.VisionModel.StdBias, m.VisionModel.StdScale)
	return []input.Multimodal{{Tensor: visionOutputs}}, nil
}

func gemma4PixelSizeFromGrid(gridTHW []int, numChannels, pixelLen int) (imgW, imgH int, err error) {
	if len(gridTHW) != 3 {
		return 0, 0, fmt.Errorf("processor_output on gemma4 requires image_grid_thw [1,H,W] pixel size")
	}
	if gridTHW[0] <= 0 {
		return 0, 0, fmt.Errorf("image_grid_thw T must be positive, got %d", gridTHW[0])
	}
	if gridTHW[0] != 1 {
		return 0, 0, fmt.Errorf("gemma4 processor_output supports T=1 per item, got T=%d", gridTHW[0])
	}
	imgH = gridTHW[1]
	imgW = gridTHW[2]
	if imgH <= 0 || imgW <= 0 {
		return 0, 0, fmt.Errorf("image_grid_thw H and W must be positive, got %v", gridTHW)
	}
	want := numChannels * imgH * imgW
	if pixelLen != want {
		return 0, 0, fmt.Errorf("processor_output pixel_values length %d != %d (channels=%d H=%d W=%d)", pixelLen, want, numChannels, imgH, imgW)
	}
	return imgW, imgH, nil
}
