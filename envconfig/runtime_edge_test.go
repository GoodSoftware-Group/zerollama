//go:build edge

package envconfig

import "testing"

func TestRuntimeEmbedDisabledOnEdgeBuildTag(t *testing.T) {
	t.Setenv("ZEROLLAMA_RUNTIME_EMBED", "1")
	t.Setenv("ZEROLLAMA_EDGE", "0")
	if RuntimeEmbedEnabled() {
		t.Fatal("edge build must not enable runtime embed even when ZEROLLAMA_EDGE=0")
	}
}

func TestRuntimeDarwinSidecarDisabledOnEdgeBuildTag(t *testing.T) {
	t.Setenv("ZEROLLAMA_RUNTIME_DARWIN_SIDECAR", "1")
	t.Setenv("ZEROLLAMA_EDGE", "0")
	if RuntimeDarwinSidecarLikely() {
		t.Fatal("edge build must not start Darwin runtime sidecar even when ZEROLLAMA_EDGE=0")
	}
}
