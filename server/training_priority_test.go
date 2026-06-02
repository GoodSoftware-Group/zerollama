package server

import "testing"

func TestParseTrainingPriority(t *testing.T) {
	cases := []struct {
		in   string
		want TrainingPriority
	}{
		{"", TrainingPriorityNormal},
		{"low", TrainingPriorityLow},
		{"batch", TrainingPriorityLow},
		{"high", TrainingPriorityHigh},
		{"interactive", TrainingPriorityHigh},
	}
	for _, tc := range cases {
		if got := parseTrainingPriority(tc.in); got != tc.want {
			t.Fatalf("%q: got %v want %v", tc.in, got, tc.want)
		}
	}
}
