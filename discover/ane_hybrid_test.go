package discover

import "testing"

func TestHybridFromProxy(t *testing.T) {
	proxy := ANEModelProxyDims{
		Tag:             "eliza-1-2b:latest",
		EmbeddingLength: 2048,
		ProxyChannels:   256,
		ProxySpatial:    16,
	}
	entry := hybridFromProxy(proxy)
	if entry.Tag != proxy.Tag || entry.ProxyChannels != 256 {
		t.Fatalf("hybridFromProxy = %+v", entry)
	}
	ch, sp := DraftANEProxyDims(2048)
	if ch != 256 || sp != 16 {
		t.Fatalf("DraftANEProxyDims fallback = (%d,%d)", ch, sp)
	}
}
