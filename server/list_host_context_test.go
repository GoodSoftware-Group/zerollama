package server

import "testing"

func TestSuggestHostCtxFromSize(t *testing.T) {
	t.Parallel()
	const train = 131072
	weights := uint64(20 << 30) // 20 GiB

	if got := suggestHostCtxFromSize(weights, 0, train); got != 0 {
		t.Fatalf("no budget: got %d", got)
	}
	if got := suggestHostCtxFromSize(weights, weights, train); got != 0 {
		t.Fatalf("budget==weights: got %d (want 0)", got)
	}

	// Plenty of leftover → full train
	plenty := weights + weights // margin + full kv proxy
	if got := suggestHostCtxFromSize(weights, plenty, train); got != train {
		t.Fatalf("plenty: got %d want %d", got, train)
	}

	// Partial leftover → between 512 and train
	partial := weights + weights/4 // margin leaves ~0.15*weights? wait margin is 1.1*w
	// leftover = budget - 1.1*w; use budget = 1.1*w + w/4
	partial = uint64(float64(weights)*1.10) + weights/4
	got := suggestHostCtxFromSize(weights, partial, train)
	if got < 512 || got >= train {
		t.Fatalf("partial: got %d want in [512,%d)", got, train)
	}
}

func TestListContextSummaryHelpers(t *testing.T) {
	// Exercised via cmd tests; keep server package focused on size heuristic.
	t.Parallel()
	if suggestHostCtxFromSize(1<<30, 1<<30+512, 8192) == 0 {
		t.Fatal("tiny leftover should still yield min ctx")
	}
}
