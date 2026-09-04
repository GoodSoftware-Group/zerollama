package api

import "testing"

func TestAppendOutputBudgetGuidance(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "hi"}}
	if AppendOutputBudgetGuidance(msgs, 0)[0].Content != "hi" {
		t.Fatal("unset")
	}
	if AppendOutputBudgetGuidance([]Message{{Role: "user", Content: "hi"}}, -1)[0].Content != "hi" {
		t.Fatal("unlimited")
	}
	if AppendOutputBudgetGuidance([]Message{{Role: "user", Content: "hi"}}, OutputBudgetTightThreshold)[0].Content != "hi" {
		t.Fatal("at threshold")
	}
	got := AppendOutputBudgetGuidance([]Message{{Role: "user", Content: "hi"}}, 256)
	if got[0].Content != "hi\n\n"+OutputBudgetGuidance {
		t.Fatalf("got %q", got[0].Content)
	}
	again := AppendOutputBudgetGuidance(got, 256)
	if again[0].Content != got[0].Content {
		t.Fatal("not idempotent")
	}
}

func TestNumPredictFromMap(t *testing.T) {
	if NumPredictFromMap(nil) != 0 {
		t.Fatal("nil")
	}
	if NumPredictFromMap(map[string]any{"num_predict": 64}) != 64 {
		t.Fatal("int")
	}
	if NumPredictFromMap(map[string]any{"num_predict": float64(32)}) != 32 {
		t.Fatal("float64")
	}
}
