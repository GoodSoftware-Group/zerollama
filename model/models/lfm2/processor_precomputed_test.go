package lfm2

import (
	"testing"
)

func TestLfm2PixelSizeFromGrid(t *testing.T) {
	w, h, err := lfm2PixelSizeFromGrid([]int{1, 64, 64}, 3, 3*64*64)
	if err != nil || w != 64 || h != 64 {
		t.Fatalf("got w=%d h=%d err=%v", w, h, err)
	}
	_, _, err = lfm2PixelSizeFromGrid([]int{1, 64, 64}, 3, 100)
	if err == nil {
		t.Fatal("expected length mismatch")
	}
}

func TestLfm2SplitProcessorTiles_multiWithThumbnail(t *testing.T) {
	const channels, tileSize, patchSize = 3, 8, 2
	rows, cols := 1, 2
	tileElems := channels * tileSize * tileSize
	thumbW, thumbH := 4, 4
	thumbElems := channels * thumbW * thumbH
	pixels := make([]float32, rows*cols*tileElems+thumbElems)
	for i := range pixels {
		pixels[i] = float32(i)
	}

	tiles, layout, err := lfm2SplitProcessorTiles(pixels, []int{1, rows, cols}, channels, tileSize, patchSize, true)
	if err != nil {
		t.Fatalf("split: %v", err)
	}
	if layout.rows != 1 || layout.cols != 2 || !layout.hasThumbnail {
		t.Fatalf("layout=%+v", layout)
	}
	if len(tiles) != 3 {
		t.Fatalf("tiles=%d want 3", len(tiles))
	}
	if tiles[0].row != 1 || tiles[0].col != 1 || tiles[0].thumbnail {
		t.Fatalf("tile0=%+v", tiles[0])
	}
	if tiles[1].row != 1 || tiles[1].col != 2 {
		t.Fatalf("tile1=%+v", tiles[1])
	}
	if !tiles[2].thumbnail || tiles[2].w != thumbW || tiles[2].h != thumbH {
		t.Fatalf("thumb=%+v", tiles[2])
	}
	if tiles[0].pixels[0] != 0 || tiles[1].pixels[0] != float32(tileElems) {
		t.Fatalf("tile packing offsets wrong")
	}
}

func TestLfm2SplitProcessorTiles_short(t *testing.T) {
	_, _, err := lfm2SplitProcessorTiles(make([]float32, 10), []int{1, 2, 2}, 3, 8, 2, true)
	if err == nil {
		t.Fatal("expected short buffer error")
	}
}

func TestLfm2PrecomputedChunkPlan(t *testing.T) {
	n, layout, err := lfm2PrecomputedChunkPlan(8, 2, 2)
	if err != nil || n != 4 || layout.hasThumbnail {
		t.Fatalf("n=%d layout=%+v err=%v", n, layout, err)
	}
	n, layout, err = lfm2PrecomputedChunkPlan(10, 2, 2)
	if err != nil || n != 5 || !layout.hasThumbnail {
		t.Fatalf("thumb n=%d layout=%+v err=%v", n, layout, err)
	}
	_, _, err = lfm2PrecomputedChunkPlan(7, 2, 2)
	if err == nil {
		t.Fatal("expected non-divisible error")
	}
}

func TestLfm2InferThumbnailSize(t *testing.T) {
	w, h, err := lfm2InferThumbnailSize(3*16*16, 3, 16)
	if err != nil || w != 16 || h != 16 {
		t.Fatalf("square: w=%d h=%d err=%v", w, h, err)
	}
	w, h, err = lfm2InferThumbnailSize(3*32*16, 3, 16)
	if err != nil || w != 32 || h != 16 {
		t.Fatalf("rect: w=%d h=%d err=%v", w, h, err)
	}
}
