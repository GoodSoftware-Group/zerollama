package discover

import "testing"

func TestDraftANEProxyDimsEliza(t *testing.T) {
	ch, sp := DraftANEProxyDims(2048)
	if ch != 256 || sp != 16 {
		t.Fatalf("DraftANEProxyDims(2048) = (%d,%d)", ch, sp)
	}
}

func TestPrefersDraftInventory(t *testing.T) {
	if !prefersDraftInventory("eliza-1-2b-dflash") {
		t.Fatal("expected dflash preference")
	}
	if prefersDraftInventory("qwen3.6") {
		t.Fatal("qwen3.6 should use model inventory first")
	}
}
