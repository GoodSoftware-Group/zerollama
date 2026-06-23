package ollamarunner

import "testing"

func TestLfm2ImageBlockStart(t *testing.T) {
	slots := Lfm2VisionTokens{Start: 100, End: 200, UseBlock: true}
	block := []int{100, 1, 2, 200}
	if !isLfm2ImageBlockStart(block, 0, slots) {
		t.Fatal("expected block start")
	}
	if isLfm2ImageBlockStart([]int{100, 1, 2}, 0, slots) {
		t.Fatal("missing end should not start")
	}
}

func TestLfm2ImageBlockEndIndex(t *testing.T) {
	block := []int{100, 1, 2, 200, 42}
	if got := lfm2ImageBlockEndIndex(block, 0, 200); got != 3 {
		t.Fatalf("end=%d want 3", got)
	}
}

func TestSkipImageTokenRun(t *testing.T) {
	tokens := []int{396, 396, 396, 1, 2}
	if got := skipImageTokenRun(tokens, 0, 396); got != 2 {
		t.Fatalf("skip=%d want 2", got)
	}
	if !isFirstImageTokenInRun(tokens, 0, 396) {
		t.Fatal("first in run")
	}
	if isFirstImageTokenInRun(tokens, 1, 396) {
		t.Fatal("not first in run")
	}
}
