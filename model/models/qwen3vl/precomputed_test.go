package qwen3vl

import "testing"

func TestGridFromPrecomputedTHW(t *testing.T) {
	g, err := gridFromPrecomputedTHW([]int{1, 24, 32})
	if err != nil {
		t.Fatal(err)
	}
	if g.Temporal != 1 || g.Height != 24 || g.Width != 32 {
		t.Fatalf("grid=%+v", g)
	}
}

func TestGridFromPrecomputedTHW_requiresTHW(t *testing.T) {
	if _, err := gridFromPrecomputedTHW([]int{1, 2}); err == nil {
		t.Fatal("expected error")
	}
}
