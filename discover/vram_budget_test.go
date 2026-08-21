package discover

import (
	"testing"

	"github.com/ollama/ollama/ml"
)

func TestApplyVRAMBudget(t *testing.T) {
	t.Setenv("ZEROLLAMA_VRAM_BUDGET", "50%")
	devs := []ml.DeviceInfo{{
		DeviceID:    ml.DeviceID{ID: "0", Library: "CUDA"},
		TotalMemory: 8 << 30,
		FreeMemory:  8 << 30,
	}}
	got := applyVRAMBudget(devs)
	if got[0].TotalMemory != 4<<30 || got[0].FreeMemory != 4<<30 {
		t.Fatalf("got total=%d free=%d", got[0].TotalMemory, got[0].FreeMemory)
	}
}

func TestApplyVRAMBudgetUnset(t *testing.T) {
	t.Setenv("ZEROLLAMA_VRAM_BUDGET", "")
	devs := []ml.DeviceInfo{{TotalMemory: 8 << 30, FreeMemory: 7 << 30}}
	got := applyVRAMBudget(devs)
	if got[0].TotalMemory != 8<<30 || got[0].FreeMemory != 7<<30 {
		t.Fatal("unset must pass through")
	}
}
