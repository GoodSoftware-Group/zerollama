package lfm2

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

// MultimodalFromProcessorOutput runs the vision tower on HF processor pixel_values.
//
// Contracts (image_grid_thw [1,H,W]):
//   - Pixel canvas: H*W*C == len(pixel_values) — single smart-resized image (historical).
//   - Tile grid: H,W are tile rows/cols; pixel_values is row-major concat of tileSize×tileSize
//     tiles (+ optional thumbnail remainder when use_thumbnail). Matches EncodeMultimodal order.
func (m *Model) MultimodalFromProcessorOutput(ctx ml.Context, pixelValues []float32, gridTHW []int) ([]input.Multimodal, error) {
	if m.VisionModel == nil || m.VisionProjector == nil || len(m.VisionModel.Layers) == 0 {
		return nil, fmt.Errorf("lfm2: no vision model for processor_output")
	}
	if len(gridTHW) != 3 {
		return nil, fmt.Errorf("processor_output on lfm2 requires image_grid_thw [1,H,W]")
	}
	if gridTHW[0] != 1 {
		return nil, fmt.Errorf("lfm2 processor_output supports T=1, got T=%d", gridTHW[0])
	}

	channels := m.ImageProcessor.numChannels
	if channels <= 0 {
		channels = 3
	}
	patchSize := m.ImageProcessor.patchSize
	if patchSize <= 0 {
		return nil, fmt.Errorf("lfm2: invalid vision patch size")
	}

	// Prefer historical single-canvas pixel size when length matches exactly.
	if imgW, imgH, err := lfm2PixelSizeFromGrid(gridTHW, channels, len(pixelValues)); err == nil {
		return m.forwardProcessorTiles(ctx, []lfm2ProcessorTile{{
			pixels: pixelValues,
			w:      imgW,
			h:      imgH,
		}}, &visionEmbeddingLayout{rows: 1, cols: 1}, patchSize)
	}

	tiles, layout, err := lfm2SplitProcessorTiles(pixelValues, gridTHW, channels, m.ImageProcessor.tileSize, patchSize, m.ImageProcessor.useThumbnail)
	if err != nil {
		return nil, err
	}
	return m.forwardProcessorTiles(ctx, tiles, layout, patchSize)
}

type lfm2ProcessorTile struct {
	pixels    []float32
	w, h      int
	row, col  int
	thumbnail bool
}

func (m *Model) forwardProcessorTiles(ctx ml.Context, tiles []lfm2ProcessorTile, layout *visionEmbeddingLayout, patchSize int) ([]input.Multimodal, error) {
	mm := make([]input.Multimodal, 0, len(tiles))
	for i, tile := range tiles {
		if tile.w%patchSize != 0 || tile.h%patchSize != 0 {
			return nil, fmt.Errorf("processor_output pixel size %dx%d must be divisible by patch_size %d", tile.w, tile.h, patchSize)
		}
		patches := visionPatchGrid{
			Width:  tile.w / patchSize,
			Height: tile.h / patchSize,
		}
		if patches.Width == 0 || patches.Height == 0 {
			return nil, fmt.Errorf("lfm2 processor_output invalid patch grid for size %dx%d", tile.w, tile.h)
		}
		pixelTensor := ctx.Input().FromFloats(tile.pixels, tile.w, tile.h, m.ImageProcessor.numChannels)
		visionOutputs := m.VisionModel.Forward(ctx, pixelTensor, patches)
		projected := m.VisionProjector.Forward(ctx, visionOutputs, patches, m.projectorOptions)
		chunk := visionChunkData{
			tokens:    projected.Dim(1),
			row:       tile.row,
			col:       tile.col,
			thumbnail: tile.thumbnail,
		}
		if i == 0 {
			chunk.layout = layout
		}
		mm = append(mm, input.Multimodal{Tensor: projected, Data: chunk})
	}
	return mm, nil
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

// lfm2SplitProcessorTiles unpacks row-major tileSize² tiles (+ optional thumbnail).
// image_grid_thw is [1, rows, cols] for the tile grid (not pixel size).
func lfm2SplitProcessorTiles(pixelValues []float32, gridTHW []int, channels, tileSize, patchSize int, useThumbnail bool) ([]lfm2ProcessorTile, *visionEmbeddingLayout, error) {
	if len(gridTHW) != 3 || gridTHW[0] != 1 {
		return nil, nil, fmt.Errorf("lfm2 multi-tile processor_output requires image_grid_thw [1,rows,cols]")
	}
	rows, cols := gridTHW[1], gridTHW[2]
	if rows <= 0 || cols <= 0 {
		return nil, nil, fmt.Errorf("image_grid_thw tile dims must be positive, got %v", gridTHW)
	}
	if tileSize <= 0 {
		return nil, nil, fmt.Errorf("lfm2: invalid tile_size")
	}
	if channels <= 0 {
		channels = 3
	}

	nTiles := rows * cols
	tileElems := channels * tileSize * tileSize
	need := nTiles * tileElems
	if len(pixelValues) < need {
		return nil, nil, fmt.Errorf("processor_output pixel_values length %d < %d for %dx%d tiles of %d²", len(pixelValues), need, rows, cols, tileSize)
	}

	layout := &visionEmbeddingLayout{rows: rows, cols: cols}
	out := make([]lfm2ProcessorTile, 0, nTiles+1)
	off := 0
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out = append(out, lfm2ProcessorTile{
				pixels: pixelValues[off : off+tileElems],
				w:      tileSize,
				h:      tileSize,
				row:    r + 1,
				col:    c + 1,
			})
			off += tileElems
		}
	}

	rem := len(pixelValues) - off
	if rem == 0 {
		return out, layout, nil
	}
	if !useThumbnail || nTiles == 1 {
		return nil, nil, fmt.Errorf("processor_output leftover %d floats after %d tiles (thumbnail unexpected)", rem, nTiles)
	}
	tw, th, err := lfm2InferThumbnailSize(rem, channels, patchSize)
	if err != nil {
		return nil, nil, err
	}
	layout.hasThumbnail = true
	out = append(out, lfm2ProcessorTile{
		pixels:    pixelValues[off:],
		w:         tw,
		h:         th,
		thumbnail: true,
	})
	return out, layout, nil
}

func lfm2InferThumbnailSize(rem, channels, patchSize int) (w, h int, err error) {
	if channels <= 0 || rem%channels != 0 {
		return 0, 0, fmt.Errorf("thumbnail remainder %d not divisible by channels %d", rem, channels)
	}
	pixels := rem / channels
	side := int(math.Sqrt(float64(pixels)))
	if side > 0 && side*side == pixels && (patchSize <= 0 || side%patchSize == 0) {
		return side, side, nil
	}
	bestW, bestH := 0, 0
	bestDiff := math.MaxInt
	for th := 1; th*th <= pixels; th++ {
		if pixels%th != 0 {
			continue
		}
		tw := pixels / th
		if patchSize > 0 && (tw%patchSize != 0 || th%patchSize != 0) {
			continue
		}
		diff := tw - th
		if diff < 0 {
			diff = -diff
		}
		if diff < bestDiff {
			bestDiff = diff
			bestW, bestH = tw, th
		}
	}
	if bestW == 0 {
		return 0, 0, fmt.Errorf("cannot infer thumbnail HxW from %d pixels (patch_size=%d)", pixels, patchSize)
	}
	return bestW, bestH, nil
}
