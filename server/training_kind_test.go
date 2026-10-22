package server

import "testing"

func TestNormalizeTrainingJobKind(t *testing.T) {
	cases := []struct {
		in   string
		want string
		ok   bool
	}{
		{"", "train", true},
		{"train", "train", true},
		{"SFT", "train", true},
		{"run_script", "run_script", true},
		{"grpo", "grpo", true},
		{"RL", "grpo", true},
		{"unsloth", "", false},
		{"video", "", false},
	}
	for _, tc := range cases {
		got, err := normalizeTrainingJobKind(tc.in)
		if tc.ok {
			if err != nil || got != tc.want {
				t.Fatalf("in=%q: got (%q, %v) want (%q, nil)", tc.in, got, err, tc.want)
			}
		} else if err == nil {
			t.Fatalf("in=%q: expected error, got %q", tc.in, got)
		}
	}
}
