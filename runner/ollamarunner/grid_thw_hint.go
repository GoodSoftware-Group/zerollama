package ollamarunner

import (
	"log/slog"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/model/input"
)

const defaultVisionSpatialMerge = 2

type visionGridHintStats struct {
	Hinted     int
	Matched    int
	Mismatched int
}

func visionTokensFromGridTHW(grid []int, spatialMerge int) int {
	if len(grid) != 3 || grid[0] <= 0 || grid[1] <= 0 || grid[2] <= 0 {
		return 0
	}
	if spatialMerge <= 0 {
		spatialMerge = defaultVisionSpatialMerge
	}
	merge := spatialMerge * spatialMerge
	return (grid[0] * grid[1] * grid[2]) / merge
}

func visionEmbedCountsFromInputs(inputs []*input.Input) []int {
	var counts []int
	for _, inp := range inputs {
		if inp.Multimodal == nil {
			continue
		}
		counts = append(counts, ollamaVisionEmbedTokenCount(inp))
	}
	return counts
}

func ollamaVisionEmbedTokenCount(inp *input.Input) int {
	if inp.SameBatch > 0 {
		return inp.SameBatch
	}
	if len(inp.Multimodal) > 0 && inp.Multimodal[0].Tensor != nil {
		return inp.Multimodal[0].Tensor.Dim(1)
	}
	return 1
}

func logVisionGridHint(imageID int, gridTHW []int, embedTokens int) visionGridHintStats {
	if len(gridTHW) != 3 {
		return visionGridHintStats{}
	}
	stats := visionGridHintStats{Hinted: 1}
	estimate := visionTokensFromGridTHW(gridTHW, defaultVisionSpatialMerge)
	if estimate <= 0 || embedTokens <= 0 {
		slog.Debug("vision grid hint",
			"image_id", imageID,
			"grid_thw", gridTHW,
			"engine_embed_tokens", embedTokens,
			"engine", "ollama",
		)
		return stats
	}
	if estimate != embedTokens {
		stats.Mismatched = 1
		slog.Debug("vision grid hint mismatch (engine uses pixel-derived layout)",
			"image_id", imageID,
			"grid_thw", gridTHW,
			"hint_tokens", estimate,
			"engine_embed_tokens", embedTokens,
			"engine", "ollama",
		)
		return stats
	}
	stats.Matched = 1
	slog.Debug("vision grid hint match",
		"image_id", imageID,
		"grid_thw", gridTHW,
		"embed_tokens", embedTokens,
		"engine", "ollama",
	)
	return stats
}

func logVisionGridHintsFromInputs(images []llm.ImageData, inputs []*input.Input) {
	counts := visionEmbedCountsFromInputs(inputs)
	var stats visionGridHintStats
	for i, img := range images {
		embed := 0
		if i < len(counts) {
			embed = counts[i]
		}
		st := logVisionGridHint(img.ID, img.GridTHW, embed)
		stats.Hinted += st.Hinted
		stats.Matched += st.Matched
		stats.Mismatched += st.Mismatched
	}
	logVisionGridHintSummary(stats)
}

func logVisionGridHintSummary(stats visionGridHintStats) {
	if stats.Hinted == 0 {
		return
	}
	slog.Info("vision grid hints",
		"hinted", stats.Hinted,
		"matched", stats.Matched,
		"mismatched", stats.Mismatched,
		"engine", "ollama",
	)
}
