//go:build !edge

package envconfig

import "testing"

func TestEdgeBuildTagDefault(t *testing.T) {
	if EdgeBuildTag {
		t.Fatal("default build should not set EdgeBuildTag")
	}
}
