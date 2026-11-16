package siglip

import (
	"bytes"
	"fmt"
	"image"
	"math"

	"golang.org/x/image/draw"

	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"

	_ "golang.org/x/image/webp"
)

// imagePlan is the per-request geometry EncodeMedia consumes.
type imagePlan struct {
	// Pixels is NHWC float32 in the tower dtype's value range, already
	// normalized. Conv towers consume it directly. NaFlex towers consume
	// Patches instead.
	Pixels []float32
	Height int
	Width  int

	// Patches is [numPatches, patchSize*patchSize*channels], channels last
	// inside each patch, matching Siglip2ImageProcessor.
	Patches   []float32
	GridH     int
	GridW     int
	PatchSize int
}

// imageSizeForMaxPatches is transformers' get_image_size_for_max_num_patches.
// The returned height and width are multiples of patchSize, and the patch
// count is the largest scale that still fits in maxPatches.
func imageSizeForMaxPatches(height, width, patchSize, maxPatches int) (int, int) {
	const eps = 1e-5
	scaled := func(scale float64, size int) int {
		n := math.Ceil(float64(size)*scale/float64(patchSize)) * float64(patchSize)
		if n < float64(patchSize) {
			n = float64(patchSize)
		}
		return int(n)
	}
	scaleMin, scaleMax := eps/10, 100.0
	for scaleMax-scaleMin >= eps {
		scale := (scaleMin + scaleMax) / 2
		th, tw := scaled(scale, height), scaled(scale, width)
		patches := (float64(th) / float64(patchSize)) * (float64(tw) / float64(patchSize))
		if patches <= float64(maxPatches) {
			scaleMin = scale
		} else {
			scaleMax = scale
		}
	}
	return scaled(scaleMin, height), scaled(scaleMin, width)
}

func prepareImage(data []byte, v *VisionConfig, fixedH, fixedW, maxPatches int, bilinear bool) (imagePlan, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return imagePlan{}, fmt.Errorf("decode image: %w", err)
	}
	srcW, srcH := img.Bounds().Dx(), img.Bounds().Dy()
	if srcW == 0 || srcH == 0 {
		return imagePlan{}, fmt.Errorf("image has no pixels")
	}
	targetH, targetW := fixedH, fixedW
	if targetH <= 0 || targetW <= 0 {
		if maxPatches <= 0 {
			maxPatches = int(v.NumPatches)
		}
		targetH, targetW = imageSizeForMaxPatches(srcH, srcW, int(v.PatchSize), maxPatches)
	}
	if targetH%int(v.PatchSize) != 0 || targetW%int(v.PatchSize) != 0 {
		return imagePlan{}, fmt.Errorf("image size %dx%d is not divisible by patch %d", targetH, targetW, v.PatchSize)
	}

	pixels := resizeNormalized(img, targetW, targetH, int(v.NumChannels), bilinear)
	plan := imagePlan{
		Pixels:    pixels,
		Height:    targetH,
		Width:     targetW,
		GridH:     targetH / int(v.PatchSize),
		GridW:     targetW / int(v.PatchSize),
		PatchSize: int(v.PatchSize),
	}
	if v.NaFlex {
		plan.Patches = patchify(pixels, targetH, targetW, int(v.PatchSize), int(v.NumChannels))
	}
	return plan, nil
}

// NormalizeImage decodes an image, resizes it to width x height, and applies
// the SigLIP mean=std=0.5 normalization. bicubic selects Catmull-Rom; otherwise
// the resize is bilinear. channels is 1 or 3.
func NormalizeImage(data []byte, width, height, channels int, bicubic bool) ([]float32, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}
	if img.Bounds().Dx() == 0 || img.Bounds().Dy() == 0 {
		return nil, fmt.Errorf("image has no pixels")
	}
	if channels < 1 {
		channels = 3
	}
	return resizeNormalized(img, width, height, channels, !bicubic), nil
}

// Patchify matches convert_image_to_patches: each patch is stored as
// (patch, patch, channel) in row-major patch order.
func Patchify(pixels []float32, height, width, patch, channels int) []float32 {
	return patchify(pixels, height, width, patch, channels)
}

func resizeNormalized(img image.Image, targetW, targetH, channels int, bilinear bool) []float32 {
	dst := image.NewNRGBA(image.Rect(0, 0, targetW, targetH))
	scaler := draw.CatmullRom
	if bilinear {
		scaler = draw.BiLinear
	}
	scaler.Scale(dst, dst.Bounds(), img, img.Bounds(), draw.Src, nil)

	pixels := make([]float32, targetH*targetW*channels)
	for y := 0; y < targetH; y++ {
		for x := 0; x < targetW; x++ {
			p := dst.NRGBAAt(x, y)
			base := (y*targetW + x) * channels
			// Checkpoints ship mean=std=0.5, so (pixel/255 - 0.5) / 0.5.
			pixels[base] = float32(p.R)/127.5 - 1
			if channels > 1 {
				pixels[base+1] = float32(p.G)/127.5 - 1
			}
			if channels > 2 {
				pixels[base+2] = float32(p.B)/127.5 - 1
			}
		}
	}
	return pixels
}

// patchify matches convert_image_to_patches: each patch is stored as
// (patch, patch, channel) in row-major patch order.
func patchify(pixels []float32, height, width, patch, channels int) []float32 {
	gh, gw := height/patch, width/patch
	out := make([]float32, gh*gw*patch*patch*channels)
	pos := 0
	for ph := 0; ph < gh; ph++ {
		for pw := 0; pw < gw; pw++ {
			for py := 0; py < patch; py++ {
				for px := 0; px < patch; px++ {
					y := ph*patch + py
					x := pw*patch + px
					src := (y*width + x) * channels
					copy(out[pos:pos+channels], pixels[src:src+channels])
					pos += channels
				}
			}
		}
	}
	return out
}

// bilinearPosWeights samples a square position grid onto gridH x gridW with
// PyTorch's bilinear align_corners=False coordinates. Each output cell stores
// four source indices and weights. Same-size grids stay exact. Downsampling
// does not apply the antialiased triangle kernel newer PyTorch interpolate
// uses, so NaFlex position tables can differ slightly from Transformers.
func bilinearPosWeights(srcSide, gridH, gridW int) (idx []int32, wt []float32) {
	n := gridH * gridW
	idx = make([]int32, n*4)
	wt = make([]float32, n*4)
	scaleY := float64(srcSide) / float64(gridH)
	scaleX := float64(srcSide) / float64(gridW)
	clamp := func(v int) int {
		if v < 0 {
			return 0
		}
		if v >= srcSide {
			return srcSide - 1
		}
		return v
	}
	for oy := 0; oy < gridH; oy++ {
		sy := (float64(oy)+0.5)*scaleY - 0.5
		y0 := int(math.Floor(sy))
		wy := float32(sy - float64(y0))
		y0c, y1c := clamp(y0), clamp(y0+1)
		for ox := 0; ox < gridW; ox++ {
			sx := (float64(ox)+0.5)*scaleX - 0.5
			x0 := int(math.Floor(sx))
			wx := float32(sx - float64(x0))
			x0c, x1c := clamp(x0), clamp(x0+1)
			base := (oy*gridW + ox) * 4
			idx[base] = int32(y0c*srcSide + x0c)
			idx[base+1] = int32(y0c*srcSide + x1c)
			idx[base+2] = int32(y1c*srcSide + x0c)
			idx[base+3] = int32(y1c*srcSide + x1c)
			wt[base] = (1 - wy) * (1 - wx)
			wt[base+1] = (1 - wy) * wx
			wt[base+2] = wy * (1 - wx)
			wt[base+3] = wy * wx
		}
	}
	return idx, wt
}
