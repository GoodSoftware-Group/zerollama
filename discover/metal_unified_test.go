package discover

import "testing"

func TestCapMetalUnifiedFree(t *testing.T) {
	if got := capMetalUnifiedFree(100, 60); got != 60 {
		t.Fatalf("capMetalUnifiedFree(100, 60) = %d, want 60", got)
	}
	if got := capMetalUnifiedFree(100, 200); got != 100 {
		t.Fatalf("capMetalUnifiedFree(100, 200) = %d, want 100", got)
	}
	if got := capMetalUnifiedFree(0, 200); got != 200 {
		t.Fatalf("capMetalUnifiedFree(0, 200) = %d, want 200", got)
	}
	if got := capMetalUnifiedFree(100, 0); got != 0 {
		t.Fatalf("capMetalUnifiedFree(100, 0) = %d, want 0", got)
	}
}
