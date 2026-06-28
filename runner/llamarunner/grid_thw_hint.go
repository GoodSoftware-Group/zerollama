package llamarunner

import "log/slog"

const defaultVisionSpatialMerge = 2

type visionGridHintStats struct {
	Hinted     int
	Matched    int
	Mismatched int
}

func countMtmdEmbedTokens(chunks []visionChunk) int {
	n := 0
	for _, c := range chunks {
		if len(c.embed) != 0 {
			n++
		}
	}
	return n
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

// logVisionGridHint compares client grid_thw hints with mtmd embed count after encode.
func logVisionGridHint(imageID int, gridTHW []int, chunks []visionChunk) visionGridHintStats {
	if len(gridTHW) != 3 {
		return visionGridHintStats{}
	}
	stats := visionGridHintStats{Hinted: 1}
	estimate := visionTokensFromGridTHW(gridTHW, defaultVisionSpatialMerge)
	got := countMtmdEmbedTokens(chunks)
	if estimate <= 0 || got <= 0 {
		slog.Debug("vision grid hint",
			"image_id", imageID,
			"grid_thw", gridTHW,
			"mtmd_embed_tokens", got,
		)
		return stats
	}
	if estimate != got {
		stats.Mismatched = 1
		slog.Debug("vision grid hint mismatch",
			"image_id", imageID,
			"grid_thw", gridTHW,
			"hint_tokens", estimate,
			"mtmd_embed_tokens", got,
		)
		return stats
	}
	stats.Matched = 1
	slog.Info("vision grid hint match",
		"image_id", imageID,
		"grid_thw", gridTHW,
		"embed_tokens", got,
	)
	return stats
}

func logVisionGridHintSummary(stats visionGridHintStats) {
	if stats.Hinted == 0 {
		return
	}
	slog.Info("vision grid hints",
		"hinted", stats.Hinted,
		"matched", stats.Matched,
		"mismatched", stats.Mismatched,
	)
}
