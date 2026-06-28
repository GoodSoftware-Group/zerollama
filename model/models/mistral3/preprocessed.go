package mistral3

import (
	"fmt"
	"image"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

// MultimodalFromPrecomputed builds per-row vision tensors for Mistral3/Pixtral padded inject.
// Each feature row is one [IMG]…[IMG] strip (hidden × patch_width); optional grid_thw [1,H,W]
// validates row count and patch width in patch-grid units.
func (m *Model) MultimodalFromPrecomputed(ctx ml.Context, rows [][]float32, gridTHW []int) ([]input.Multimodal, error) {
	if len(rows) == 0 {
		return nil, fmt.Errorf("precomputed feature is empty")
	}
	hidden := len(rows[0])
	if hidden == 0 {
		return nil, fmt.Errorf("precomputed feature rows must be non-empty")
	}

	mm := make([]input.Multimodal, 0, len(rows))
	for i, row := range rows {
		if len(row)%hidden != 0 {
			return nil, fmt.Errorf("precomputed row %d length %d not divisible by hidden %d", i, len(row), hidden)
		}
		width := len(row) / hidden
		tensor := ctx.Input().FromFloats(row, hidden, width)
		mm = append(mm, input.Multimodal{Tensor: tensor})
	}

	if len(gridTHW) == 3 {
		if gridTHW[0] != 1 {
			return nil, fmt.Errorf("mistral3 precomputed grid_thw T must be 1, got %d", gridTHW[0])
		}
		if gridTHW[1] != len(rows) {
			return nil, fmt.Errorf("mistral3 precomputed grid_thw H=%d != row count %d", gridTHW[1], len(rows))
		}
		patchW := len(rows[0]) / hidden
		if gridTHW[2] != patchW {
			return nil, fmt.Errorf("mistral3 precomputed grid_thw W=%d != patch width %d", gridTHW[2], patchW)
		}
	}
	return mm, nil
}

// MultimodalFromProcessorOutput runs the vision tower on HF processor pixel_values.
// image_grid_thw is [1, pixel_height, pixel_width] (channel-first pixels).
func (m *Model) MultimodalFromProcessorOutput(ctx ml.Context, pixelValues []float32, gridTHW []int) ([]input.Multimodal, error) {
	size, err := mistral3PixelSizeFromGrid(gridTHW, m.ImageProcessor.numChannels, len(pixelValues))
	if err != nil {
		return nil, err
	}
	if size.X%m.ImageProcessor.patchSize != 0 || size.Y%m.ImageProcessor.patchSize != 0 {
		return nil, fmt.Errorf("processor_output pixel size %dx%d must be divisible by patch_size %d", size.X, size.Y, m.ImageProcessor.patchSize)
	}

	pixelTensor := ctx.Input().FromFloats(pixelValues, size.X, size.Y, m.ImageProcessor.numChannels)
	visionOutputs := m.VisionModel.Forward(ctx, pixelTensor)
	features, patchGrid := m.MultiModalProjector.Forward(ctx, visionOutputs, size)

	out := make([]input.Multimodal, patchGrid.Y)
	for i := range out {
		out[i].Tensor = features.View(ctx, features.Stride(1)*patchGrid.X*i, features.Dim(0), features.Stride(1), patchGrid.X)
	}
	return out, nil
}

func mistral3PixelSizeFromGrid(gridTHW []int, numChannels, pixelLen int) (image.Point, error) {
	if len(gridTHW) != 3 {
		return image.Point{}, fmt.Errorf("processor_output on mistral3 requires image_grid_thw [1,H,W] pixel size")
	}
	if gridTHW[0] != 1 {
		return image.Point{}, fmt.Errorf("mistral3 processor_output supports T=1, got T=%d", gridTHW[0])
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
