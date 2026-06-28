package lfm2

import (
	"fmt"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

// MultimodalFromPrecomputed builds vision tensors from SGLang precomputed_embedding rows.
// Single-tile only: one projected chunk. Multi-tile LFM2 layouts still require PNG encode.
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
	if len(gridTHW) == 3 && (gridTHW[1] > 1 || gridTHW[2] > 1) {
		return nil, fmt.Errorf("lfm2 precomputed_embedding supports single-tile grid only; got grid_thw %v", gridTHW)
	}

	flat := make([]float32, 0, hidden*len(rows))
	for _, row := range rows {
		flat = append(flat, row...)
	}
	tensor := ctx.Input().FromFloats(flat, hidden, len(rows))
	chunk := visionChunkData{tokens: len(rows), layout: &visionEmbeddingLayout{rows: 1, cols: 1}}
	return []input.Multimodal{{Tensor: tensor, Data: chunk}}, nil
}
