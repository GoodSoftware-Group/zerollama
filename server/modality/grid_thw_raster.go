package modality

import "github.com/ollama/ollama/api"

// GridTHWPerRaster returns optional [1,H,W] per entry in msg.Images (parallel slice).
// Still-image rasters get nil; each video frame inherits H,W from its VideoSpans clip grid when GridTHWExplicit.
func GridTHWPerRaster(msg api.Message) [][]int {
	n := len(msg.Images)
	if n == 0 {
		return nil
	}
	out := make([][]int, n)

	videoFrames := 0
	for _, sp := range msg.VideoSpans {
		videoFrames += sp.FrameCount
	}
	still := len(msg.Images) - videoFrames
	if still < 0 {
		still = 0
	}

	frameIdx := still
	for _, sp := range msg.VideoSpans {
		if len(sp.GridTHW) != 3 || sp.GridTHW[1] <= 0 || sp.GridTHW[2] <= 0 || !sp.GridTHWExplicit {
			frameIdx += sp.FrameCount
			continue
		}
		h, w := sp.GridTHW[1], sp.GridTHW[2]
		for f := 0; f < sp.FrameCount; f++ {
			if frameIdx < n {
				out[frameIdx] = []int{1, h, w}
			}
			frameIdx++
		}
	}
	return out
}

// GridTHWHasHints reports whether any raster carries a non-empty grid hint.
func GridTHWHasHints(grids [][]int) bool {
	for _, g := range grids {
		if len(g) == 3 {
			return true
		}
	}
	return false
}
