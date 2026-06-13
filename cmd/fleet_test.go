package cmd

import "testing"

func TestPortFromListen(t *testing.T) {
	tests := []struct {
		listen string
		want   int
	}{
		{"0.0.0.0:11450", 11450},
		{":11450", 11450},
		{"127.0.0.1:11450", 11450},
	}
	for _, tc := range tests {
		got, err := portFromListen(tc.listen)
		if err != nil {
			t.Fatalf("listen=%q: %v", tc.listen, err)
		}
		if got != tc.want {
			t.Fatalf("listen=%q got=%d want=%d", tc.listen, got, tc.want)
		}
	}
}
