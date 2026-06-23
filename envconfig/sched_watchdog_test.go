package envconfig

import (
	"testing"
	"time"
)

func TestMemoryReclaimThreshold(t *testing.T) {
	t.Setenv("ZEROLLAMA_MEMORY_RECLAIM_THRESHOLD", "0.95")
	if got := MemoryReclaimThreshold(); got != 0.95 {
		t.Fatalf("got %v", got)
	}
	t.Setenv("ZEROLLAMA_MEMORY_RECLAIM_THRESHOLD", "bad")
	if got := MemoryReclaimThreshold(); got != 0 {
		t.Fatalf("invalid should disable, got %v", got)
	}
}

func TestRunnerBusyTimeout(t *testing.T) {
	t.Setenv("ZEROLLAMA_RUNNER_BUSY_TIMEOUT", "10m")
	if got := RunnerBusyTimeout(); got != 10*time.Minute {
		t.Fatalf("got %v", got)
	}
}
