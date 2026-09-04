package mlxrunner

import "testing"

func TestGenerationIsLoopShortCycle(t *testing.T) {
	t.Parallel()
	cycle := []int32{10, 11, 12, 13} // ": 2,5" class
	var toks []int32
	for range 6 {
		toks = append(toks, cycle...)
	}
	if !generationIsLoop(toks) {
		t.Fatal("6× 4-token cycle should stop")
	}
	three := append(append([]int32{}, cycle...), cycle...)
	three = append(three, cycle...)
	if generationIsLoop(three) {
		t.Fatal("3× 4-token cycle is under the short-period repeat bar")
	}
}

func TestGenerationIsLoopTripleCycle(t *testing.T) {
	t.Parallel()
	cycle := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	var toks []int32
	for range 3 {
		toks = append(toks, cycle...)
	}
	if !generationIsLoop(toks) {
		t.Fatal("triple 8-token cycle should stop")
	}
}

func TestGenerationIsLoopNeedsThree(t *testing.T) {
	t.Parallel()
	cycle := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	toks := append(append([]int32{}, cycle...), cycle...)
	if generationIsLoop(toks) {
		t.Fatal("two copies is not a loop")
	}
}

func TestGenerationIsLoopNovel(t *testing.T) {
	t.Parallel()
	toks := make([]int32, 64)
	for i := range toks {
		toks[i] = int32(i)
	}
	if generationIsLoop(toks) {
		t.Fatal("unique tail is not a loop")
	}
}

func TestGenerationIsLoopLongPeriodTenCopies(t *testing.T) {
	t.Parallel()
	cycle := make([]int32, 50)
	for i := range cycle {
		cycle[i] = int32(i + 1)
	}
	var toks []int32
	for range 10 {
		toks = append(toks, cycle...)
	}
	if !generationIsLoop(toks) {
		t.Fatal("10× 50-token cycle should stop (mlx-serve long-period tier)")
	}
	three := append(append([]int32{}, cycle...), cycle...)
	three = append(three, cycle...)
	if generationIsLoop(three) {
		t.Fatal("3× 50-token cycle is under the long-period repeat bar")
	}
}

func TestLoopSpanStart(t *testing.T) {
	t.Parallel()
	cycle := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	prefix := []int32{9, 10}
	var toks []int32
	toks = append(toks, prefix...)
	for range 3 {
		toks = append(toks, cycle...)
	}
	start, ok := loopSpanStart(toks)
	if !ok || start != 2 {
		t.Fatalf("start=%d ok=%v want 2", start, ok)
	}
}

func TestGenerationIsNearRepeatLoopCollapsedTail(t *testing.T) {
	t.Parallel()
	toks := make([]int32, 128)
	for i := range toks {
		toks[i] = int32(i % 3)
	}
	if !generationIsNearRepeatLoop(toks) {
		t.Fatal("low-vocab repeating tail should stop")
	}
}

func TestGenerationIsNearRepeatLoopNovel(t *testing.T) {
	t.Parallel()
	toks := make([]int32, 128)
	for i := range toks {
		toks[i] = int32(i)
	}
	if generationIsNearRepeatLoop(toks) {
		t.Fatal("unique tail is not a near-repeat")
	}
}
