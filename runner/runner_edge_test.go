//go:build edge

package runner

import (
	"strings"
	"testing"
)

func TestExecuteEdgeStub(t *testing.T) {
	err := Execute([]string{"runner", "--model", "/tmp/m.gguf"})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "not included in edge builds") {
		t.Fatalf("err=%v", err)
	}
}
