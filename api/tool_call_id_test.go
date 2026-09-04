package api

import "testing"

func TestEnsureToolCallID(t *testing.T) {
	if got := EnsureToolCallID("call_abc", 3); got != "call_abc" {
		t.Fatalf("kept id: %q", got)
	}
	if got := EnsureToolCallID("  ", 2); got != "call_2" {
		t.Fatalf("empty id: %q", got)
	}
	if got := EnsureToolCallID("", -1); got != "call_0" {
		t.Fatalf("negative index: %q", got)
	}
}
