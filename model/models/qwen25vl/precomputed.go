package qwen25vl

import (
	"fmt"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

// MultimodalFromPrecomputed builds vision tensors from SGLang precomputed_embedding rows.
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
	grid, err := gridFromPrecomputedTHW(gridTHW)
	if err != nil {
		return nil, err
	}

	flat := make([]float32, 0, hidden*len(rows))
	for _, row := range rows {
		flat = append(flat, row...)
	}
	tensor := ctx.Input().FromFloats(flat, hidden, len(rows))
	return []input.Multimodal{{Tensor: tensor, Data: grid}}, nil
}

func gridFromPrecomputedTHW(gridTHW []int) (*Grid, error) {
	if len(gridTHW) != 3 {
		return nil, fmt.Errorf("precomputed_embedding on ollama-engine requires grid_thw [T,H,W]")
	}
	if gridTHW[0] <= 0 || gridTHW[1] <= 0 || gridTHW[2] <= 0 {
		return nil, fmt.Errorf("grid_thw values must be positive, got %v", gridTHW)
	}
	return &Grid{Temporal: gridTHW[0], Height: gridTHW[1], Width: gridTHW[2]}, nil
}
