package llama4

import "testing"

func TestLlama4PixelSizeFromGrid(t *testing.T) {
	p, err := llama4PixelSizeFromGrid([]int{1, 336, 336}, 3, 3*336*336)
	if err != nil || p.X != 336 || p.Y != 336 {
		t.Fatalf("got %+v err=%v", p, err)
	}
	_, err = llama4PixelSizeFromGrid([]int{1, 336, 336}, 3, 100)
	if err == nil {
		t.Fatal("expected length mismatch")
	}
}

func TestLlama4SplitProcessorPixels_single(t *testing.T) {
	const imageSize, channels = 336, 3
	local := make([]float32, channels*imageSize*imageSize)
	gotLocal, gotGlobal, size, err := llama4SplitProcessorPixels(local, []int{1, imageSize, imageSize}, channels, imageSize)
	if err != nil {
		t.Fatalf("split: %v", err)
	}
	if size.X != imageSize || size.Y != imageSize || len(gotLocal) != len(local) || gotGlobal != nil {
		t.Fatalf("size=%+v local=%d global=%d", size, len(gotLocal), len(gotGlobal))
	}
}

func TestLlama4SplitProcessorPixels_multiRequiresGlobal(t *testing.T) {
	const imageSize, channels = 336, 3
	h, w := imageSize*2, imageSize*2
	localOnly := make([]float32, channels*h*w)
	_, _, _, err := llama4SplitProcessorPixels(localOnly, []int{1, h, w}, channels, imageSize)
	if err == nil {
		t.Fatal("expected multi-tile without global to fail")
	}

	full := make([]float32, channels*h*w+channels*imageSize*imageSize)
	local, global, size, err := llama4SplitProcessorPixels(full, []int{1, h, w}, channels, imageSize)
	if err != nil {
		t.Fatalf("split: %v", err)
	}
	if size.X != w || size.Y != h {
		t.Fatalf("size=%+v", size)
	}
	if len(local) != channels*h*w || len(global) != channels*imageSize*imageSize {
		t.Fatalf("local=%d global=%d", len(local), len(global))
	}
}

func TestLlama4SplitProcessorPixels_singleRejectsGlobal(t *testing.T) {
	const imageSize, channels = 336, 3
	full := make([]float32, channels*imageSize*imageSize*2)
	_, _, _, err := llama4SplitProcessorPixels(full, []int{1, imageSize, imageSize}, channels, imageSize)
	if err == nil {
		t.Fatal("expected single-tile+global to fail")
	}
}

func TestLlama4SplitProcessorPixels_notDivisible(t *testing.T) {
	_, _, _, err := llama4SplitProcessorPixels(make([]float32, 3*100*100), []int{1, 100, 100}, 3, 336)
	if err == nil {
		t.Fatal("expected not divisible by image_size")
	}
}
