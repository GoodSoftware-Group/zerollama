package mllama

import (
	"fmt"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

// MultimodalFromPrecomputed builds vision tensors from SGLang precomputed_embedding rows
// (post-projector cross-attention states, same layout as mtmd embed chunks on ggml llamarunner).
func (m *Model) MultimodalFromPrecomputed(ctx ml.Context, rows [][]float32, _ []int) ([]input.Multimodal, error) {
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

	flat := make([]float32, 0, hidden*len(rows))
	for _, row := range rows {
		flat = append(flat, row...)
	}
	tensor := ctx.Input().FromFloats(flat, hidden, len(rows))
	return []input.Multimodal{{Tensor: tensor}}, nil
}
