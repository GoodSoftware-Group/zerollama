//go:build edge

package envconfig

import "testing"

func TestGgmlRunnerLinkedEdge(t *testing.T) {
	if GgmlRunnerLinked() {
		t.Fatal("expected ggml runner unlinked in edge build")
	}
}
