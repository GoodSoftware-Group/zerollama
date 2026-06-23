//go:build linux

package discover

import (
	"testing"

	"github.com/ollama/ollama/format"
)

func TestApplyCgroupMemoryLimit_unlimitedKeepsMemAvailable(t *testing.T) {
	memAvailable := uint64(22 * format.GibiByte)
	mem := memInfo{
		TotalMemory: uint64(24 * format.GibiByte),
		FreeMemory:  memAvailable,
	}
	got := applyCgroupMemoryLimit(mem, 0, false)
	if got.FreeMemory != memAvailable {
		t.Fatalf("FreeMemory = %d, want MemAvailable %d", got.FreeMemory, memAvailable)
	}
}

func TestApplyCgroupMemoryLimit_finiteSetsTotalKeepsMemAvailable(t *testing.T) {
	memAvailable := uint64(20 * format.GibiByte)
	limit := uint64(8 * format.GibiByte)
	mem := memInfo{
		TotalMemory: uint64(24 * format.GibiByte),
		FreeMemory:  memAvailable,
	}
	got := applyCgroupMemoryLimit(mem, limit, true)
	if got.FreeMemory != memAvailable {
		t.Fatalf("FreeMemory = %d, want MemAvailable %d", got.FreeMemory, memAvailable)
	}
	if got.TotalMemory != limit {
		t.Fatalf("TotalMemory = %d, want %d", got.TotalMemory, limit)
	}
}
