package modality

import (
	"fmt"

	"github.com/ollama/ollama/api"
)

const defaultSpatialMergeSize = 2

// VisionTokensFromGridTHW estimates Qwen/SGLang vision placeholder count from [T,H,W] patch grid.
// Formula: T×H×W / spatial_merge² (same as img_grid_thw.prod() // merge² in SGLang qwen_vl.py).
func VisionTokensFromGridTHW(grid []int, spatialMergeSize int) int {
	if len(grid) != 3 || grid[0] <= 0 || grid[1] <= 0 || grid[2] <= 0 {
		return 0
	}
	if spatialMergeSize <= 0 {
		spatialMergeSize = defaultSpatialMergeSize
	}
	merge := spatialMergeSize * spatialMergeSize
	return (grid[0] * grid[1] * grid[2]) / merge
}

func validateVideoSpanGridTHW(sp api.VideoSpan) error {
	if len(sp.GridTHW) == 0 {
		return nil
	}
	if len(sp.GridTHW) != 3 {
		return fmt.Errorf("video_spans grid_thw must be [T,H,W], got len %d", len(sp.GridTHW))
	}
	for i, v := range sp.GridTHW {
		if v <= 0 {
			return fmt.Errorf("video_spans grid_thw[%d] must be positive, got %d", i, v)
		}
	}
	if sp.FrameCount > 0 && sp.GridTHW[0] != sp.FrameCount {
		return fmt.Errorf("video_spans grid_thw[0]=%d must match frame_count=%d", sp.GridTHW[0], sp.FrameCount)
	}
	return nil
}

func videoSpanVisionTokens(sp api.VideoSpan, tokensPerImage, spatialMergeSize int) int {
	if n := VisionTokensFromGridTHW(sp.GridTHW, spatialMergeSize); n > 0 {
		return n
	}
	if sp.FrameCount <= 0 {
		return 0
	}
	if tokensPerImage <= 0 {
		tokensPerImage = 768
	}
	return sp.FrameCount * tokensPerImage
}
