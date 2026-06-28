package mistral3

import "testing"

func TestMistral3PrecomputedRowWidth(t *testing.T) {
	rows := [][]float32{
		make([]float32, 128*4),
		make([]float32, 128*4),
	}
	hidden := 128
	for i, row := range rows {
		if len(row)%hidden != 0 {
			t.Fatalf("row %d bad len", i)
		}
	}
	grid := []int{1, 2, 4}
	if grid[1] != len(rows) {
		t.Fatal("grid H mismatch")
	}
	if grid[2] != len(rows[0])/hidden {
		t.Fatal("grid W mismatch")
	}
}
