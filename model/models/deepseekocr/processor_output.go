package deepseekocr

import (
	"fmt"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

const (
	deepseekOCRTileSize = 640
	deepseekOCRBaseSize = 1024
	deepseekOCRMinTiles = 2
	deepseekOCRMaxTiles = 9
	deepseekOCRChannels = 3
)

var _ model.ProcessorOutputMultimodalIngest = (*Model)(nil)

// MultimodalFromProcessorOutput runs SAM + CLIP + projector on HF-normalized pixels.
//
// image_grid_thw is [1, tile_rows, tile_cols] for the local crop grid (2 ≤ rows*cols ≤ 9).
// pixel_values packs row-major 640²×3 local tiles followed by one 1024²×3 global canvas —
// the same layout EncodeMultimodal builds after ProcessImage.
func (m *Model) MultimodalFromProcessorOutput(ctx ml.Context, pixelValues []float32, gridTHW []int) ([]input.Multimodal, error) {
	if m.Sam == nil || m.Vision == nil || m.Projector == nil {
		return nil, model.ErrNoVisionModel
	}
	rows, cols, localElems, _, crop, err := deepseekOCRValidateProcessorPixels(pixelValues, gridTHW)
	if err != nil {
		return nil, err
	}
	blocks := rows * cols
	patches := ctx.Input().FromFloats(pixelValues[:localElems], deepseekOCRTileSize, deepseekOCRTileSize, deepseekOCRChannels, blocks)
	original := ctx.Input().FromFloats(pixelValues[localElems:], deepseekOCRBaseSize, deepseekOCRBaseSize, deepseekOCRChannels)
	return m.multimodalFromPatches(ctx, patches, original, crop)
}

func deepseekOCRValidateProcessorPixels(pixelValues []float32, gridTHW []int) (rows, cols, localElems, globalElems int, crop []int, err error) {
	if len(gridTHW) != 3 {
		return 0, 0, 0, 0, nil, fmt.Errorf("processor_output on deepseekocr requires image_grid_thw [1,rows,cols]")
	}
	if gridTHW[0] != 1 {
		return 0, 0, 0, 0, nil, fmt.Errorf("deepseekocr processor_output supports T=1, got T=%d", gridTHW[0])
	}
	rows, cols = gridTHW[1], gridTHW[2]
	if rows <= 0 || cols <= 0 {
		return 0, 0, 0, 0, nil, fmt.Errorf("image_grid_thw tile dims must be positive, got %v", gridTHW)
	}
	blocks := rows * cols
	if blocks < deepseekOCRMinTiles || blocks > deepseekOCRMaxTiles {
		return 0, 0, 0, 0, nil, fmt.Errorf("deepseekocr tile count %d out of range [%d,%d]", blocks, deepseekOCRMinTiles, deepseekOCRMaxTiles)
	}

	localElems = blocks * deepseekOCRChannels * deepseekOCRTileSize * deepseekOCRTileSize
	globalElems = deepseekOCRChannels * deepseekOCRBaseSize * deepseekOCRBaseSize
	want := localElems + globalElems
	if len(pixelValues) != want {
		return 0, 0, 0, 0, nil, fmt.Errorf(
			"processor_output pixel_values length %d != %d (%dx%d tiles of %d² + global %d²)",
			len(pixelValues), want, rows, cols, deepseekOCRTileSize, deepseekOCRBaseSize,
		)
	}

	// ProcessImage crop is [x,y] = [cols, rows].
	crop = []int{cols, rows}
	return rows, cols, localElems, globalElems, crop, nil
}
