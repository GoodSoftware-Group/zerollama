package qwen25vl

import (
	"image"
	"testing"
)

func TestResizeFromGridHint(t *testing.T) {
	h, w, ok := resizeFromGridHint(14, 2, []int{1, 24, 32})
	if !ok || h != 24*14 || w != 32*14 {
		t.Fatalf("got %dx%d ok=%v", h, w, ok)
	}
	if _, _, ok := resizeFromGridHint(14, 2, []int{1, 23, 32}); ok {
		t.Fatal("odd H should fail merge alignment")
	}
	if _, _, ok := resizeFromGridHint(14, 2, nil); ok {
		t.Fatal("nil grid should fail")
	}
}

func TestProcessImage_gridHint(t *testing.T) {
	p := ImageProcessor{
		numChannels:       3,
		patchSize:         14,
		temporalPatchSize: 2,
		mergeSize:         2,
		minPixels:         56 * 56,
		maxPixels:         2 << 20,
		factor:            28,
		imageMean:         [3]float32{0.5, 0.5, 0.5},
		imageStd:          [3]float32{0.5, 0.5, 0.5},
	}
	img := image.NewRGBA(image.Rect(0, 0, 200, 150))
	_, grid, err := p.ProcessImage(img, []int{1, 4, 6})
	if err != nil {
		t.Fatal(err)
	}
	if grid.Height != 4 || grid.Width != 6 {
		t.Fatalf("grid=%+v want H=4 W=6", grid)
	}
}
