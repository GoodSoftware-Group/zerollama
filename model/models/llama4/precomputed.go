package llama4

import (
	"fmt"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

// MultimodalFromPrecomputed builds vision tensors from SGLang precomputed_embedding rows.
// grid_thw [1, tile_h, tile_w] is optional Llama4 tile grid (patch tiles, not pixels); omit or
// [1,1,1] for a single global chunk. Multi-tile layouts append a final global chunk (+1).
func (m *Model) MultimodalFromPrecomputed(ctx ml.Context, rows [][]float32, gridTHW []int) ([]input.Multimodal, error) {
	if len(rows) == 0 {
		return nil, fmt.Errorf("precomputed feature is empty")
	}
	hidden := len(rows[0])
	if hidden == 0 {
		return nil, fmt.Errorf("precomputed feature rows must be non-empty")
	}
	for i, row := range rows {
		if len(row) != hidden {
			return nil, fmt.Errorf("precomputed row %d width %d != %d", i, len(row), hidden)
		}
	}

	tileH, tileW := 1, 1
	if len(gridTHW) == 3 {
		if gridTHW[0] != 1 {
			return nil, fmt.Errorf("llama4 precomputed grid_thw T must be 1, got %d", gridTHW[0])
		}
		if gridTHW[1] <= 0 || gridTHW[2] <= 0 {
			return nil, fmt.Errorf("llama4 precomputed grid_thw tile dims must be positive, got %v", gridTHW)
		}
		tileH, tileW = gridTHW[1], gridTHW[2]
	}

	numTiles := tileH * tileW
	numChunks := 1
	if numTiles > 1 {
		numChunks = numTiles + 1
	}
	if len(rows)%numChunks != 0 {
		return nil, fmt.Errorf("precomputed rows %d not divisible into %d llama4 chunks (grid %dx%d)", len(rows), numChunks, tileH, tileW)
	}
	rowsPerChunk := len(rows) / numChunks

	if numChunks == 1 {
		flat := flattenRows(rows)
		tensor := ctx.Input().FromFloats(flat, hidden, len(rows))
		return []input.Multimodal{{Tensor: tensor, Data: &separator{}}}, nil
	}

	var mm []input.Multimodal
	chunkIdx := 0
	for y := 0; y < tileH; y++ {
		for x := 0; x < tileW; x++ {
			chunkRows := rows[chunkIdx*rowsPerChunk : (chunkIdx+1)*rowsPerChunk]
			tensor := ctx.Input().FromFloats(flattenRows(chunkRows), hidden, len(chunkRows))
			sep := &separator{}
			if x < tileW-1 {
				sep.x = true
			} else if y < tileH-1 {
				sep.y = true
			}
			mm = append(mm, input.Multimodal{Tensor: tensor, Data: sep})
			chunkIdx++
		}
	}
	globalRows := rows[chunkIdx*rowsPerChunk:]
	tensor := ctx.Input().FromFloats(flattenRows(globalRows), hidden, len(globalRows))
	mm = append(mm, input.Multimodal{Tensor: tensor, Data: &separator{}})
	return mm, nil
}

func flattenRows(rows [][]float32) []float32 {
	if len(rows) == 0 {
		return nil
	}
	hidden := len(rows[0])
	flat := make([]float32, 0, hidden*len(rows))
	for _, row := range rows {
		flat = append(flat, row...)
	}
	return flat
}
