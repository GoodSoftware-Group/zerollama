package discover

import "testing"

func TestPrefillMatmulGFLOP(t *testing.T) {
	got := PrefillMatmulGFLOP(256, 256, 512)
	want := 2.0 * 256 * 256 * 512 / 1e9
	if got != want {
		t.Fatalf("PrefillMatmulGFLOP = %v want %v", got, want)
	}
}

func TestPrefillProxyFromEmbed(t *testing.T) {
	ic, oc, seq := PrefillProxyFromEmbed(2048, 8192)
	if ic != 2048 || oc != 2048 || seq != 4096 {
		t.Fatalf("PrefillProxyFromEmbed = (%d,%d,%d)", ic, oc, seq)
	}
}

func TestPrefillProxyFromEmbedCapFull(t *testing.T) {
	ic, oc, seq := PrefillProxyFromEmbedCap(5120, 256, 0)
	if ic != 5120 || oc != 5120 || seq != 256 {
		t.Fatalf("PrefillProxyFromEmbedCap full = (%d,%d,%d)", ic, oc, seq)
	}
	ic, oc, _ = PrefillProxyFromEmbedCap(5120, 256, DefaultPrefillICCap)
	if ic != 2048 || oc != 2048 {
		t.Fatalf("PrefillProxyFromEmbedCap capped = (%d,%d)", ic, oc)
	}
}

func TestPrefillExpertOC(t *testing.T) {
	if got := PrefillExpertOC(2048); got != 512 {
		t.Fatalf("PrefillExpertOC(2048)=%d want 512", got)
	}
	if got := PrefillExpertOC(128); got != 64 {
		t.Fatalf("PrefillExpertOC(128)=%d want 64", got)
	}
}
