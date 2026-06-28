package llamarunner

import "testing"

func TestVisionChunksFromPrecomputed(t *testing.T) {
	rows := [][]float32{{1, 2}, {3, 4}}
	vc := visionChunksFromPrecomputed(rows)
	if len(vc) != 2 || len(vc[0].embed) != 2 || vc[1].embed[0] != 3 {
		t.Fatalf("got %+v", vc)
	}
}

func TestVisionChunksFromPrecomputed_skipsEmptyRows(t *testing.T) {
	vc := visionChunksFromPrecomputed([][]float32{{1}, {}, {2}})
	if len(vc) != 2 {
		t.Fatalf("got %d chunks", len(vc))
	}
}
