package lfm2vl

import "testing"

func publishedLimits() imageLimits {
	return imageLimits{
		downsample: 2,
		patch:      16,
		minTokens:  64,
		maxTokens:  256,
		tile:       512,
		minTiles:   2,
		maxTiles:   10,
		tolerance:  2,
		split:      true,
		thumbnail:  true,
		bicubic:    true,
	}
}

func TestLayoutSmallImageStaysOneTile(t *testing.T) {
	lay, err := layoutImage(100, 100, publishedLimits())
	if err != nil {
		t.Fatal(err)
	}
	if lay.gridW != 1 || lay.gridH != 1 || len(lay.tiles) != 1 {
		t.Fatalf("layout = %+v, want one tile", lay)
	}
	if lay.tiles[0].width != 256 || lay.tiles[0].height != 256 {
		t.Fatalf("tile = %dx%d, want 256x256", lay.tiles[0].width, lay.tiles[0].height)
	}
	n, err := tokensForTile(lay.tiles[0], publishedLimits())
	if err != nil {
		t.Fatal(err)
	}
	if n != 64 {
		t.Fatalf("tokens = %d, want 64", n)
	}
}

func TestLayoutLargeImageTilesAndThumbnail(t *testing.T) {
	lay, err := layoutImage(2000, 2000, publishedLimits())
	if err != nil {
		t.Fatal(err)
	}
	if lay.gridW != 3 || lay.gridH != 3 {
		t.Fatalf("grid = %dx%d, want 3x3", lay.gridW, lay.gridH)
	}
	if len(lay.tiles) != 10 || !lay.tiles[9].thumbnail {
		t.Fatalf("tiles = %+v, want nine tiles plus a thumbnail", lay.tiles)
	}
	if lay.tiles[9].width != 512 || lay.tiles[9].height != 512 {
		t.Fatalf("thumbnail = %dx%d, want 512x512", lay.tiles[9].width, lay.tiles[9].height)
	}
	lim := publishedLimits()
	total := 0
	for _, tile := range lay.tiles {
		n, err := tokensForTile(tile, lim)
		if err != nil {
			t.Fatal(err)
		}
		if n != 256 {
			t.Fatalf("tile tokens = %d, want 256", n)
		}
		total += n
	}
	if total != 2560 {
		t.Fatalf("image tokens = %d, want 2560", total)
	}
}

func TestRoundByFactorHalfEven(t *testing.T) {
	if got := roundByFactor(2000, 32); got != 1984 {
		t.Fatalf("roundByFactor(2000, 32) = %d, want 1984", got)
	}
}
