// Package size resolves image dimensions and aspect ratios for diffusion models.
package size

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Supported aspect ratio presets (long edge fits within MaxSide).
var supportedAspects = map[string][2]int{
	"16:9": {16, 9},
	"9:16": {9, 16},
	"3:2":  {3, 2},
	"2:3":  {2, 3},
	"1:1":  {1, 1},
}

// SupportedAspects returns preset names for help text.
func SupportedAspects() []string {
	return []string{"16:9", "9:16", "3:2", "2:3", "1:1"}
}

// MaxSide returns the maximum long edge for image generation on this host.
//
// WHY gpuAvailable matters: the Go serve process does not load MLX, so callers in
// routes.go must not resolve final dimensions — only the MLX runner subprocess passes
// mlx.GPUIsAvailable() here. On CUDA hosts we default to 384 because denoise
// activations dominate VRAM at 1024² even when ~12GB of weights fit.
//
// ZEROLLAMA_IMAGE_MAX_SIDE overrides the default. CUDA_VISIBLE_DEVICES and
// OLLAMA_LIBRARY_PATH are fallback hints when the parent process probes size
// before the runner starts (e.g. CLI help text).
func MaxSide(gpuAvailable bool) int32 {
	if v := os.Getenv("ZEROLLAMA_IMAGE_MAX_SIDE"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 64 {
			return Round8(int32(n))
		}
	}
	if gpuAvailable || os.Getenv("CUDA_VISIBLE_DEVICES") != "" || os.Getenv("OLLAMA_LIBRARY_PATH") != "" {
		return 384
	}
	return 1024
}

// ParseAspect returns width:height ratio parts for a preset name.
func ParseAspect(name string) (int, int, bool) {
	name = strings.TrimSpace(strings.ToLower(name))
	if name == "" {
		return 0, 0, false
	}
	name = strings.ReplaceAll(name, "x", ":")
	ar, ok := supportedAspects[name]
	if !ok {
		return 0, 0, false
	}
	return ar[0], ar[1], true
}

// Round8 rounds down to a multiple of 8 (minimum 64).
func Round8(n int32) int32 {
	n = (n / 8) * 8
	if n < 64 {
		return 64
	}
	return n
}

// Resolve computes width and height from optional explicit dimensions and an aspect preset.
// When aspect is set and neither dimension is set, the long edge equals maxSide.
// When aspect is set and one dimension is set, the other is derived from the ratio.
func Resolve(width, height int32, aspect string, maxSide int32) (int32, int32, error) {
	aw, ah, ok := ParseAspect(aspect)
	if aspect != "" && !ok {
		return 0, 0, fmt.Errorf(
			"unsupported aspect_ratio %q (supported: %s)",
			aspect, strings.Join(SupportedAspects(), ", "),
		)
	}

	if ok {
		switch {
		case width > 0 && height > 0:
			// Explicit box wins; aspect is ignored.
		case width > 0:
			height = Round8(int32(float64(width) * float64(ah) / float64(aw)))
		case height > 0:
			width = Round8(int32(float64(height) * float64(aw) / float64(ah)))
		default:
			if aw >= ah {
				width = maxSide
				height = Round8(int32(float64(maxSide) * float64(ah) / float64(aw)))
			} else {
				height = maxSide
				width = Round8(int32(float64(maxSide) * float64(aw) / float64(ah)))
			}
		}
	} else if width <= 0 && height <= 0 {
		width = maxSide
		height = maxSide
	} else {
		if width <= 0 {
			width = maxSide
		}
		if height <= 0 {
			height = maxSide
		}
	}

	w, h := Clamp(width, height, maxSide)
	return w, h, nil
}

// Clamp scales dimensions so the long edge fits within maxSide and rounds to multiples of 8.
func Clamp(w, h, maxSide int32) (int32, int32) {
	if w <= 0 {
		w = maxSide
	}
	if h <= 0 {
		h = maxSide
	}
	long := w
	if h > long {
		long = h
	}
	if long > maxSide {
		scale := float64(maxSide) / float64(long)
		w = int32(float64(w) * scale)
		h = int32(float64(h) * scale)
	}
	return Round8(w), Round8(h)
}
