package model

import "testing"

func TestAcceptDraftPrefix(t *testing.T) {
	if n := AcceptDraftPrefix([]int32{1, 2, 3}, []int32{1, 2, 9}); n != 2 {
		t.Fatalf("got %d", n)
	}
	if n := AcceptDraftPrefix([]int32{1}, []int32{2}); n != 0 {
		t.Fatalf("got %d", n)
	}
	if n := AcceptDraftPrefix(nil, []int32{1}); n != 0 {
		t.Fatalf("got %d", n)
	}
}

func TestArgmaxLogits(t *testing.T) {
	if got := ArgmaxLogits([]float32{0.1, 0.9, 0.2}); got != 1 {
		t.Fatalf("got %d", got)
	}
	if got := ArgmaxLogits(nil); got != -1 {
		t.Fatalf("got %d", got)
	}
}
