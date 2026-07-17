package modality

import (
	"bytes"
	"image"
	_ "image/png"
	"math"

	"github.com/ollama/ollama/api"
)

// Qwen3-VL defaults for grid_thw estimates on native ffmpeg expansion.
// Why duplicate SmartResize here: modality must not import model/models/qwen3vl (runner graph);
// this path feeds preflight/usage and M-RoPE ViT hints (GridTHWPerRaster → mtmd/ollama-engine).
const (
	qwenVLGridPatchSize     = 14
	qwenVLGridFactor        = 28 // patch_size * spatial_merge_size
	qwenVLGridLongestEdge   = 2 << 20
	qwenVLGridShortestEdge  = 64 << 10
	qwenVLGridMaxAspectRatio = 200
)

// computeVideoGridTHWFromFrames returns [T,H,W] patch grid for a sampled clip (first frame dims).
// T is frame count; H/W are patch-grid dims after Qwen-style smart resize. Nil when undecodable.
func computeVideoGridTHWFromFrames(frames []api.ImageData, policy VideoSamplingPolicy) []int {
	if len(frames) == 0 {
		return nil
	}
	height, width, err := imageDimensions(frames[0])
	if err != nil || height <= 0 || width <= 0 {
		return nil
	}
	patch := policy.visionPatchSize()
	factor := policy.visionGridFactor()
	rh, rw := qwenVLSmartResize(height, width, factor)
	if rh <= 0 || rw <= 0 {
		return nil
	}
	gridH := rh / patch
	gridW := rw / patch
	if gridH <= 0 || gridW <= 0 {
		return nil
	}
	return []int{len(frames), gridH, gridW}
}

func videoSpanFromExpand(frames []api.ImageData, cachedGrid []int, policy VideoSamplingPolicy) api.VideoSpan {
	span := api.VideoSpan{FrameCount: len(frames)}
	if len(cachedGrid) == 3 {
		span.GridTHW = append([]int(nil), cachedGrid...)
		return span
	}
	if grid := computeVideoGridTHWFromFrames(frames, policy); len(grid) == 3 {
		span.GridTHW = grid
	}
	return span
}

func imageDimensions(data []byte) (height, width int, err error) {
	cfg, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return 0, 0, err
	}
	return cfg.Height, cfg.Width, nil
}

func qwenVLSmartResize(height, width, factor int) (int, int) {
	if factor <= 0 {
		factor = qwenVLGridFactor
	}
	if height < factor {
		height = factor
	}
	if width < factor {
		width = factor
	}
	if min(height, width) == 0 {
		return 0, 0
	}
	if max(height, width)/min(height, width) > qwenVLGridMaxAspectRatio {
		return 0, 0
	}

	round := func(x float64) int { return int(math.RoundToEven(x)) }

	hBar := round(float64(height)/float64(factor)) * factor
	wBar := round(float64(width)/float64(factor)) * factor

	if hBar*wBar > qwenVLGridLongestEdge {
		beta := math.Sqrt(float64(height*width) / float64(qwenVLGridLongestEdge))
		hBar = int(math.Floor(float64(height)/beta/float64(factor))) * factor
		wBar = int(math.Floor(float64(width)/beta/float64(factor))) * factor
	} else if hBar*wBar < qwenVLGridShortestEdge {
		beta := math.Sqrt(float64(qwenVLGridShortestEdge) / float64(height*width))
		hBar = int(math.Ceil(float64(height)*beta/float64(factor))) * factor
		wBar = int(math.Ceil(float64(width)*beta/float64(factor))) * factor
	}
	if hBar <= 0 || wBar <= 0 {
		return 0, 0
	}
	return hBar, wBar
}
