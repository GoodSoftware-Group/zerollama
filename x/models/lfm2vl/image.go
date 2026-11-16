package lfm2vl

import (
	"bytes"
	"image"
	"math"
	"sort"
)

// imageLimits is the resolved LFM2-VL processor. Numbers match
// LiquidAI/LFM2-VL-1.6B when a checkpoint omits a field: 512px tiles,
// SigLIP2 patch 16, downsample 2, and a split once the image exceeds
// max_image_tokens * max_pixels_tolerance.
type imageLimits struct {
	downsample int
	patch      int
	minTokens  int
	maxTokens  int
	tile       int
	minTiles   int
	maxTiles   int
	tolerance  float64
	split      bool
	thumbnail  bool
	bicubic    bool
}

type tile struct {
	width, height int
	thumbnail     bool
}

type imageLayout struct {
	gridW, gridH int
	tiles        []tile
}

func (l imageLimits) factor() int { return l.patch * l.downsample }

func layoutImage(srcW, srcH int, lim imageLimits) (imageLayout, error) {
	if srcW < 1 || srcH < 1 {
		return imageLayout{}, errImage("image has no pixels")
	}
	if lim.patch < 1 || lim.downsample < 1 || lim.factor() < 1 {
		return imageLayout{}, errImage("invalid patch or downsample")
	}
	thumbW, thumbH := smartResize(srcW, srcH, lim)
	large := lim.split && imageTooLarge(srcW, srcH, lim)
	if !large {
		return imageLayout{
			gridW: 1, gridH: 1,
			tiles: []tile{{width: thumbW, height: thumbH}},
		}, nil
	}
	gridW, gridH := gridLayout(srcW, srcH, lim)
	tiles := make([]tile, 0, gridW*gridH+1)
	for row := 0; row < gridH; row++ {
		for col := 0; col < gridW; col++ {
			tiles = append(tiles, tile{width: lim.tile, height: lim.tile})
		}
	}
	if lim.thumbnail && gridW*gridH != 1 {
		tiles = append(tiles, tile{width: thumbW, height: thumbH, thumbnail: true})
	}
	return imageLayout{gridW: gridW, gridH: gridH, tiles: tiles}, nil
}

func imageTooLarge(width, height int, lim imageLimits) bool {
	factor := lim.factor()
	h := max(lim.patch, roundByFactor(height, factor))
	w := max(lim.patch, roundByFactor(width, factor))
	limit := float64(lim.maxTokens) * float64(lim.patch*lim.patch) * float64(lim.downsample*lim.downsample) * lim.tolerance
	return float64(h*w) > limit
}

// smartResize returns width, height. Both are multiples of patch*downsample
// and the pixel count stays inside the min/max token budget.
func smartResize(width, height int, lim imageLimits) (int, int) {
	factor := lim.factor()
	minPixels := lim.minTokens * lim.patch * lim.patch * lim.downsample * lim.downsample
	maxPixels := lim.maxTokens * lim.patch * lim.patch * lim.downsample * lim.downsample
	h := max(factor, roundByFactor(height, factor))
	w := max(factor, roundByFactor(width, factor))
	if h*w > maxPixels {
		beta := math.Sqrt(float64(height*width) / float64(maxPixels))
		h = max(factor, int(math.Floor(float64(height)/beta/float64(factor)))*factor)
		w = max(factor, int(math.Floor(float64(width)/beta/float64(factor)))*factor)
	} else if h*w < minPixels {
		beta := math.Sqrt(float64(minPixels) / float64(height*width))
		h = int(math.Ceil(float64(height)*beta/float64(factor))) * factor
		w = int(math.Ceil(float64(width)*beta/float64(factor))) * factor
	}
	return w, h
}

func gridLayout(width, height int, lim imageLimits) (int, int) {
	ratios := targetRatios(lim.minTiles, lim.maxTiles)
	aspect := float64(width) / float64(height)
	bestW, bestH := 1, 1
	bestDiff := math.Inf(1)
	area := float64(width * height)
	for _, ratio := range ratios {
		w, h := ratio[0], ratio[1]
		diff := math.Abs(aspect - float64(w)/float64(h))
		if diff < bestDiff {
			bestDiff = diff
			bestW, bestH = w, h
			continue
		}
		if diff == bestDiff {
			target := float64(lim.tile*lim.tile) * float64(w*h)
			if area > 0.5*target {
				bestW, bestH = w, h
			}
		}
	}
	return bestW, bestH
}

// targetRatios matches the processor: unique (width, height) grids whose
// tile count is inside [min, max], ordered by tile count. Equal counts keep
// the order they were first generated.
func targetRatios(minTiles, maxTiles int) [][2]int {
	seen := map[[2]int]bool{}
	ratios := make([][2]int, 0, maxTiles*maxTiles)
	for n := minTiles; n <= maxTiles; n++ {
		for w := 1; w <= n; w++ {
			for h := 1; h <= n; h++ {
				if w*h < minTiles || w*h > maxTiles {
					continue
				}
				key := [2]int{w, h}
				if seen[key] {
					continue
				}
				seen[key] = true
				ratios = append(ratios, key)
			}
		}
	}
	sort.SliceStable(ratios, func(i, j int) bool {
		return ratios[i][0]*ratios[i][1] < ratios[j][0]*ratios[j][1]
	})
	return ratios
}

func tokensForTile(t tile, lim imageLimits) (int, error) {
	if t.width%lim.patch != 0 || t.height%lim.patch != 0 {
		return 0, errImage("tile size is not divisible by the encoder patch")
	}
	gh, gw := t.height/lim.patch, t.width/lim.patch
	if gh%lim.downsample != 0 || gw%lim.downsample != 0 {
		return 0, errImage("patch grid is not divisible by downsample_factor")
	}
	return (gh / lim.downsample) * (gw / lim.downsample), nil
}

// roundByFactor is transformers' round_by_factor, including Python 3's
// half-even round.
func roundByFactor(n, factor int) int {
	if factor <= 0 {
		return n
	}
	return pyRound(float64(n)/float64(factor)) * factor
}

func pyRound(x float64) int {
	floor := math.Floor(x)
	diff := x - floor
	if diff < 0.5 {
		return int(floor)
	}
	if diff > 0.5 {
		return int(floor) + 1
	}
	if int(math.Abs(floor))%2 == 0 {
		return int(floor)
	}
	return int(floor) + 1
}

type imageError string

func (e imageError) Error() string { return string(e) }

func errImage(msg string) error { return imageError("lfm2-vl: " + msg) }

func imageConfig(data []byte) (decodedSize, error) {
	cfg, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return decodedSize{}, errImage("decode image: " + err.Error())
	}
	if cfg.Width < 1 || cfg.Height < 1 {
		return decodedSize{}, errImage("image has no pixels")
	}
	return decodedSize{w: cfg.Width, h: cfg.Height}, nil
}
