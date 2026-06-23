package ollamarunner

import "testing"

func TestIsLlama4ImageBlockStart(t *testing.T) {
	block := []int{llama4ImageBoundary, llama4PatchToken, 1, 2, llama4ImageBoundary}
	if !isLlama4ImageBlockStart(block, 0) {
		t.Fatal("expected block start")
	}
	if isLlama4ImageBlockStart(block, 4) {
		t.Fatal("closing boundary alone is not a new block")
	}
	if isLlama4ImageBlockStart([]int{llama4ImageBoundary, 1, 2}, 0) {
		t.Fatal("unclosed block without patch/image should not inject")
	}
}

func TestLlama4ImageBlockEndIndex(t *testing.T) {
	block := []int{llama4ImageBoundary, llama4ImageToken, llama4PatchToken, llama4ImageBoundary, 42}
	if got := llama4ImageBlockEndIndex(block, 0); got != 3 {
		t.Fatalf("got %d want 3", got)
	}
}
