//go:build edge

package envconfig

import "testing"

func TestEdgeBuildTagTrue(t *testing.T) {
	if !EdgeBuildTag {
		t.Fatal("expected EdgeBuildTag true with -tags edge")
	}
}
