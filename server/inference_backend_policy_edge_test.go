//go:build edge

package server

import (
	"testing"
)

func TestInferenceBackendPolicyEdgeBuildGgmlUnlinked(t *testing.T) {
	p := inferenceBackendPolicy()
	if p.GgmlLinked {
		t.Fatal("edge build should report ggml_linked false")
	}
}
