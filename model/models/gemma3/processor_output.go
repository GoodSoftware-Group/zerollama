package gemma3

import (
	"fmt"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

// MultimodalFromProcessorOutput runs the vision tower on HF processor pixel_values.
// image_grid_thw is interpreted as [1, image_size, image_size] (channel-first pixels).
func (m *Model) MultimodalFromProcessorOutput(ctx ml.Context, pixelValues []float32, gridTHW []int) ([]input.Multimodal, error) {
	size := m.ImageProcessor.imageSize
	if err := validateGemma3ProcessorPixels(gridTHW, size, m.ImageProcessor.numChannels, len(pixelValues)); err != nil {
		return nil, err
	}

	pixelTensor := ctx.Input().FromFloats(pixelValues, size, size, m.ImageProcessor.numChannels)
	visionOutputs := m.VisionModel.Forward(ctx, pixelTensor)
	visionOutputs = m.MultiModalProjector.Forward(ctx, visionOutputs, m.imageSize, m.patchSize, m.VisionModel.eps)
	return []input.Multimodal{{Tensor: visionOutputs}}, nil
}

func validateGemma3ProcessorPixels(gridTHW []int, imageSize, numChannels, pixelLen int) error {
	want := numChannels * imageSize * imageSize
	if pixelLen != want {
		return fmt.Errorf("processor_output pixel_values length %d != %d (image_size=%d)", pixelLen, want, imageSize)
	}
	if len(gridTHW) == 0 {
		return nil
	}
	if len(gridTHW) != 3 {
		return fmt.Errorf("processor_output on gemma3 requires image_grid_thw [1,%d,%d]", imageSize, imageSize)
	}
	if gridTHW[0] != 1 || gridTHW[1] != imageSize || gridTHW[2] != imageSize {
		return fmt.Errorf("gemma3 processor_output expects image_grid_thw [1,%d,%d], got %v", imageSize, imageSize, gridTHW)
	}
	return nil
}
