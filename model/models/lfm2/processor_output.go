package lfm2

import (
	"fmt"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

// MultimodalFromProcessorOutput runs the vision tower on HF processor pixel_values.
// Single-tile only: image_grid_thw is [1, pixel_height, pixel_width].
func (m *Model) MultimodalFromProcessorOutput(ctx ml.Context, pixelValues []float32, gridTHW []int) ([]input.Multimodal, error) {
	imgW, imgH, err := lfm2PixelSizeFromGrid(gridTHW, m.ImageProcessor.numChannels, len(pixelValues))
	if err != nil {
		return nil, err
	}

	patchSize := m.ImageProcessor.patchSize
	if imgW%patchSize != 0 || imgH%patchSize != 0 {
		return nil, fmt.Errorf("processor_output pixel size %dx%d must be divisible by patch_size %d", imgW, imgH, patchSize)
	}

	pixelTensor := ctx.Input().FromFloats(pixelValues, imgW, imgH, m.ImageProcessor.numChannels)
	patches := visionPatchGrid{
		Width:  imgW / patchSize,
		Height: imgH / patchSize,
	}
	if patches.Width == 0 || patches.Height == 0 {
		return nil, fmt.Errorf("lfm2 processor_output invalid patch grid for size %dx%d", imgW, imgH)
	}

	visionOutputs := m.VisionModel.Forward(ctx, pixelTensor, patches)
	projected := m.VisionProjector.Forward(ctx, visionOutputs, patches, m.projectorOptions)
	chunk := visionChunkData{
		tokens: projected.Dim(1),
		layout: &visionEmbeddingLayout{rows: 1, cols: 1},
	}
	return []input.Multimodal{{Tensor: projected, Data: chunk}}, nil
}

func lfm2PixelSizeFromGrid(gridTHW []int, numChannels, pixelLen int) (imgW, imgH int, err error) {
	if len(gridTHW) != 3 {
		return 0, 0, fmt.Errorf("processor_output on lfm2 requires image_grid_thw [1,H,W] pixel size")
	}
	if gridTHW[0] != 1 {
		return 0, 0, fmt.Errorf("lfm2 processor_output supports T=1, got T=%d", gridTHW[0])
	}
	imgH = gridTHW[1]
	imgW = gridTHW[2]
	if imgH <= 0 || imgW <= 0 {
		return 0, 0, fmt.Errorf("image_grid_thw H and W must be positive, got %v", gridTHW)
	}
	want := numChannels * imgH * imgW
	if pixelLen != want {
		return 0, 0, fmt.Errorf("processor_output pixel_values length %d != %d (H=%d W=%d)", pixelLen, want, imgH, imgW)
	}
	return imgW, imgH, nil
}
