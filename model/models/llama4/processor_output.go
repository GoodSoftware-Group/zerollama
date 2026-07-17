package llama4

import (
	"fmt"
	"image"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

// MultimodalFromProcessorOutput runs the vision tower on HF processor pixel_values.
//
// image_grid_thw is [1, H, W] of the padded *local* canvas (must be divisible by image_size).
// Multi-tile canvases (ratioW*ratioH > 1) require the global tileSize² image appended after
// the local canvas — same packing EncodeMultimodal uses after ProcessImage.
func (m *Model) MultimodalFromProcessorOutput(ctx ml.Context, pixelValues []float32, gridTHW []int) ([]input.Multimodal, error) {
	if m.VisionModel == nil || len(m.VisionModel.Layers) < 1 {
		return nil, model.ErrNoVisionModel
	}

	local, global, size, err := llama4SplitProcessorPixels(pixelValues, gridTHW, m.numChannels, m.imageSize)
	if err != nil {
		return nil, err
	}
	return m.multimodalFromPixels(ctx, local, global, size)
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

// llama4SplitProcessorPixels unpacks local canvas (+ optional global tile) from processor_output.
func llama4SplitProcessorPixels(pixelValues []float32, gridTHW []int, numChannels, imageSize int) (local, global []float32, size image.Point, err error) {
	if len(gridTHW) != 3 {
		return nil, nil, image.Point{}, fmt.Errorf("processor_output on llama4 requires image_grid_thw [1,H,W] pixel size")
	}
	if gridTHW[0] != 1 {
		return nil, nil, image.Point{}, fmt.Errorf("llama4 processor_output supports T=1, got T=%d", gridTHW[0])
	}
	if imageSize <= 0 {
		return nil, nil, image.Point{}, fmt.Errorf("llama4: invalid vision image_size")
	}
	if numChannels <= 0 {
		numChannels = 3
	}

	h, w := gridTHW[1], gridTHW[2]
	if h <= 0 || w <= 0 {
		return nil, nil, image.Point{}, fmt.Errorf("image_grid_thw H and W must be positive, got %v", gridTHW)
	}
	if h%imageSize != 0 || w%imageSize != 0 {
		return nil, nil, image.Point{}, fmt.Errorf("llama4 canvas %dx%d not divisible by image_size %d", w, h, imageSize)
	}

	size = image.Point{X: w, Y: h}
	ratioW, ratioH := w/imageSize, h/imageSize
	localElems := numChannels * h * w
	globalElems := numChannels * imageSize * imageSize
	multi := ratioW*ratioH > 1

	switch {
	case len(pixelValues) == localElems && !multi:
		return pixelValues, nil, size, nil
	case len(pixelValues) == localElems && multi:
		return nil, nil, image.Point{}, fmt.Errorf(
			"llama4 multi-tile processor_output canvas %dx%d requires global tile appended (+%d floats)",
			w, h, globalElems,
		)
	case len(pixelValues) == localElems+globalElems && multi:
		return pixelValues[:localElems], pixelValues[localElems:], size, nil
	case len(pixelValues) == localElems+globalElems && !multi:
		return nil, nil, image.Point{}, fmt.Errorf("llama4 single-tile processor_output must not include a global tile")
	default:
		want := localElems
		if multi {
			want = localElems + globalElems
		}
		return nil, nil, image.Point{}, fmt.Errorf(
			"processor_output pixel_values length %d != %d (local %dx%d%s)",
			len(pixelValues), want, w, h, map[bool]string{true: "+global", false: ""}[multi],
		)
	}
}
