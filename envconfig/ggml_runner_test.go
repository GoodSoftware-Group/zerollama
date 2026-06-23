//go:build !edge

package envconfig

import "testing"

func TestGgmlRunnerLinkedDefault(t *testing.T) {
	if !GgmlRunnerLinked() {
		t.Fatal("expected ggml runner linked in default build")
	}
}
