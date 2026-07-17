package lfm2

import (
	"fmt"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

// MultimodalFromPrecomputed builds vision tensors from SGLang precomputed_embedding rows.
//
// grid_thw:
//   - omit / empty / [1,1,1] — single projected chunk
//   - [1, rows, cols] — rows*cols equal-sized tile chunks (+ optional equal-sized thumbnail
//     when len(rows) divisible by rows*cols+1). Matches EncodeMultimodal chunk order.
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

	tileRows, tileCols := 1, 1
	if len(gridTHW) == 3 {
		if gridTHW[0] != 1 {
			return nil, fmt.Errorf("lfm2 precomputed grid_thw T must be 1, got %d", gridTHW[0])
		}
		if gridTHW[1] <= 0 || gridTHW[2] <= 0 {
			return nil, fmt.Errorf("lfm2 precomputed grid_thw tile dims must be positive, got %v", gridTHW)
		}
		tileRows, tileCols = gridTHW[1], gridTHW[2]
	}

	nTiles := tileRows * tileCols
	if nTiles == 1 {
		flat := flattenPrecomputedRows(rows)
		tensor := ctx.Input().FromFloats(flat, hidden, len(rows))
		chunk := visionChunkData{tokens: len(rows), layout: &visionEmbeddingLayout{rows: 1, cols: 1}}
		return []input.Multimodal{{Tensor: tensor, Data: chunk}}, nil
	}

	chunks, layout, err := lfm2PrecomputedChunkPlan(len(rows), tileRows, tileCols)
	if err != nil {
		return nil, err
	}
	rowsPerChunk := len(rows) / chunks
	layoutInfo := layout
	mm := make([]input.Multimodal, 0, chunks)
	for i := 0; i < chunks; i++ {
		chunkRows := rows[i*rowsPerChunk : (i+1)*rowsPerChunk]
		tensor := ctx.Input().FromFloats(flattenPrecomputedRows(chunkRows), hidden, len(chunkRows))
		cd := visionChunkData{tokens: len(chunkRows)}
		if i == 0 {
			cd.layout = layoutInfo
		}
		if i < nTiles {
			cd.row = i/tileCols + 1
			cd.col = i%tileCols + 1
		} else {
			cd.thumbnail = true
		}
		mm = append(mm, input.Multimodal{Tensor: tensor, Data: cd})
	}
	return mm, nil
}

func lfm2PrecomputedChunkPlan(rowCount, tileRows, tileCols int) (chunks int, layout *visionEmbeddingLayout, err error) {
	nTiles := tileRows * tileCols
	layout = &visionEmbeddingLayout{rows: tileRows, cols: tileCols}
	if rowCount%nTiles == 0 {
		return nTiles, layout, nil
	}
	withThumb := nTiles + 1
	if rowCount%withThumb == 0 {
		layout.hasThumbnail = true
		return withThumb, layout, nil
	}
	return 0, nil, fmt.Errorf("precomputed rows %d not divisible into %d tiles or %d tiles+thumbnail (grid %dx%d)", rowCount, nTiles, withThumb, tileRows, tileCols)
}

func flattenPrecomputedRows(rows [][]float32) []float32 {
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
