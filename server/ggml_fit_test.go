package server

import (
	"testing"

	"github.com/ollama/ollama/ml"
)

func TestGgmlWeightsFitLeavesRoomForRuntime(t *testing.T) {
	gpus := []ml.DeviceInfo{
		{FreeMemory: 17 << 30},
		{FreeMemory: 17 << 30},
	}
	if !ggmlWeightsFit(gpus, 1400<<20) {
		t.Fatal("1.4GiB weights should fit in 34GiB free")
	}
	if ggmlWeightsFit(gpus, 40<<30) {
		t.Fatal("40GiB weights should not fit")
	}
	if ggmlWeightsFit(nil, 1<<20) {
		t.Fatal("no gpus")
	}
	integrated := []ml.DeviceInfo{{FreeMemory: 64 << 30, Integrated: true}}
	if ggmlWeightsFit(integrated, 1<<20) {
		t.Fatal("unified memory must not keep the runtime resident on a weight-only fit")
	}
}
