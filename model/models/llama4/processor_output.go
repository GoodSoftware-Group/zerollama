package llama4

import (
	"fmt"
	"image"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

// MultimodalFromProcessorOutput runs the vision tower on HF processor pixel_values for a single-tile image.
// image_grid_thw is [1, pixel_height, pixel_width] of the padded local canvas (no multi-tile global leg).
func (m *Model) MultimodalFromProcessorOutput(ctx ml.Context, pixelValues []float32, gridTHW []int) ([]input.Multimodal, error) {
	size, err := llama4PixelSizeFromGrid(gridTHW, m.numChannels, len(pixelValues))
	if err != nil {
		return nil, err
	}
	if size.X/m.imageSize*size.Y/m.imageSize > 1 {
		return nil, fmt.Errorf("llama4 processor_output multi-tile canvas %dx%d not supported via preprocessed path", size.X, size.Y)
	}

	tilesLocal := ctx.Input().FromFloats(pixelValues, size.X, size.Y, m.numChannels)
	ratioW, ratioH := size.X/m.imageSize, size.Y/m.imageSize

	tilesLocal = tilesLocal.Reshape(ctx, size.X/ratioW, ratioW, size.Y, m.numChannels).Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	tilesLocal = tilesLocal.Reshape(ctx, size.X/ratioW*size.Y/ratioH, ratioH, ratioW, m.numChannels).Permute(ctx, 0, 3, 2, 1).Contiguous(ctx)
	tilesLocal = tilesLocal.Reshape(ctx, size.X/ratioW, size.Y/ratioH, m.numChannels, ratioH*ratioW)

	visionOutputs := m.VisionModel.Forward(ctx, tilesLocal)
	visionOutputs = visionOutputs.Reshape(ctx, visionOutputs.Dim(0), visionOutputs.Dim(1)*visionOutputs.Dim(2)*visionOutputs.Dim(3))
	projectedOutputs := m.Projector.Forward(ctx, visionOutputs)

	patchesPerChunk := projectedOutputs.Dim(1)
	view := projectedOutputs.Slice(ctx, 1, 0, patchesPerChunk, 1)
	return []input.Multimodal{{Tensor: view, Data: &separator{}}}, nil
}

func llama4PixelSizeFromGrid(gridTHW []int, numChannels, pixelLen int) (image.Point, error) {
	if len(gridTHW) != 3 {
		return image.Point{}, fmt.Errorf("processor_output on llama4 requires image_grid_thw [1,H,W] pixel size")
	}
	if gridTHW[0] != 1 {
		return image.Point{}, fmt.Errorf("llama4 processor_output supports T=1, got T=%d", gridTHW[0])
	}
	h, w := gridTHW[1], gridTHW[2]
	if h <= 0 || w <= 0 {
		return image.Point{}, fmt.Errorf("image_grid_thw H and W must be positive, got %v", gridTHW)
	}
	want := numChannels * h * w
	if pixelLen != want {
		return image.Point{}, fmt.Errorf("processor_output pixel_values length %d != %d (H=%d W=%d)", pixelLen, want, h, w)
	}
	return image.Point{X: w, Y: h}, nil
}
