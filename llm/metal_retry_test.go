package llm

import (
	"errors"
	"runtime"
	"testing"
)

func TestShouldRetryWithMetalTensorDisabled(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("Metal tensor retry is darwin-only")
	}

	status := NewStatusWriter(nil)
	status.SetLastError("failed to initialize metal backend")

	if !ShouldRetryWithMetalTensorDisabled(errors.New("load failed"), status) {
		t.Fatal("expected Metal tensor retry for metal backend init failure")
	}
}
