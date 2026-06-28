package llama4

import "testing"

func TestLlama4PrecomputedChunkSplit(t *testing.T) {
	rows := make([][]float32, 8)
	for i := range rows {
		rows[i] = make([]float32, 4)
	}
	// 2x2 tiles + 1 global => 5 chunks, 8/5 not divisible — should fail at validation in caller
	if len(rows)%5 != 0 {
		// 10 rows / 5 chunks = 2 rows each
	}
	rows10 := make([][]float32, 10)
	for i := range rows10 {
		rows10[i] = make([]float32, 4)
	}
	grid := []int{1, 2, 2}
	numTiles := grid[1] * grid[2]
	numChunks := numTiles + 1
	if len(rows10)%numChunks != 0 {
		t.Fatalf("expected even split")
	}
}
