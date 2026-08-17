package discover

import (
	"testing"

	"github.com/ollama/ollama/format"
)

func TestEvaluateHostMemPressure_swapAndFull(t *testing.T) {
	p := evaluateHostMemPressure(CgroupMem{
		Limit:       24 * format.GibiByte,
		HasLimit:    true,
		Anon:        23 * format.GibiByte,
		SwapCurrent: 18 * format.GibiByte,
		SwapMax:     24 * format.GibiByte,
		HasSwapMax:  true,
	}, 0.88, 0.35, format.GibiByte, true)
	if !p.Pressure {
		t.Fatal("want pressure when anon 23/24GiB and swap 18GiB")
	}
	if p.ClientMessage() == "" || p.Reason == "" {
		t.Fatal("want client-facing reason")
	}
}

func TestEvaluateHostMemPressure_plenty(t *testing.T) {
	p := evaluateHostMemPressure(CgroupMem{
		Limit:    24 * format.GibiByte,
		HasLimit: true,
		Anon:     8 * format.GibiByte,
		Current:  14 * format.GibiByte,
	}, 0.88, 0.35, format.GibiByte, true)
	if p.Pressure {
		t.Fatalf("unexpected pressure: %s", p.Reason)
	}
}

func TestEvaluateHostMemPressure_currentNearLimit(t *testing.T) {
	p := evaluateHostMemPressure(CgroupMem{
		Limit:       24 * format.GibiByte,
		HasLimit:    true,
		Anon:        1 * format.GibiByte,
		Current:     22 * format.GibiByte,
		SwapCurrent: 2 * format.GibiByte,
		SwapMax:     24 * format.GibiByte,
		HasSwapMax:  true,
	}, 0.88, 0.35, format.GibiByte, true)
	if !p.Pressure {
		t.Fatal("want pressure when cgroup current is 22/24GiB even if anon is small")
	}
}

func TestEvaluateHostMemPressure_darwinUnifiedNoSwap(t *testing.T) {
	p := evaluateHostMemPressure(CgroupMem{
		Limit:    24 * format.GibiByte,
		HasLimit: true,
		Current:  22 * format.GibiByte,
	}, 0.88, 0.35, format.GibiByte, false)
	if !p.Pressure {
		t.Fatal("darwin unified memory should pressure at 88% even without swap")
	}
}
