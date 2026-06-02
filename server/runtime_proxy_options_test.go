package server

import "testing"

func TestNumPredictFromOptions(t *testing.T) {
	t.Parallel()

	if _, ok := numPredictFromOptions(nil); ok {
		t.Fatal("expected no limit when options nil")
	}

	n, ok := numPredictFromOptions(map[string]any{"num_predict": 256})
	if !ok || n != 256 {
		t.Fatalf("got (%d, %v), want (256, true)", n, ok)
	}

	if _, ok := numPredictFromOptions(map[string]any{"num_predict": -1}); ok {
		t.Fatal("expected no limit for num_predict -1")
	}

	if _, ok := numPredictFromOptions(map[string]any{}); ok {
		t.Fatal("expected no limit for empty options")
	}
}

func TestRuntimeOptionsWithNumPredict(t *testing.T) {
	t.Parallel()

	if got := runtimeOptionsWithNumPredict(0, false); len(got) != 0 {
		t.Fatalf("unlimited: got %v", got)
	}
	if got := runtimeOptionsWithNumPredict(128, true); got["num_predict"] != 128 {
		t.Fatalf("limited: got %v", got)
	}
}
