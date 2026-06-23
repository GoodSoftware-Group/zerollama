package discover

import "testing"

func TestDraftANEProxyDims(t *testing.T) {
	ch, sp := DraftANEProxyDims(2048)
	if sp != 16 || ch != 256 {
		t.Fatalf("DraftANEProxyDims(2048) = (%d,%d), want (256,16)", ch, sp)
	}
	ch, sp = DraftANEProxyDims(0)
	if ch != 64 || sp != 16 {
		t.Fatalf("DraftANEProxyDims(0) = (%d,%d), want (64,16)", ch, sp)
	}
	ch, _ = DraftANEProxyDims(8192)
	if ch != 512 {
		t.Fatalf("DraftANEProxyDims(8192) channels = %d, want 512 cap", ch)
	}
}
