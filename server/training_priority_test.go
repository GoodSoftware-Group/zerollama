package server

import "testing"

// Thin sanity that server wrappers stay wired; full cases live in x/trainprio.
func TestParseTrainingPriorityWrapper(t *testing.T) {
	p, err := parseTrainingPriority("overnight")
	if err != nil || p != TrainingPriorityLow {
		t.Fatalf("got %v %v", p, err)
	}
	_, err = parseTrainingPriority("higgh")
	if err == nil {
		t.Fatal("expected error")
	}
}
