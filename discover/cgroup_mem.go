package discover

import (
	"fmt"

	"github.com/ollama/ollama/format"
)

// CgroupMem is a Linux cgroup v2 memory snapshot (zeros on other OS).
type CgroupMem struct {
	Limit       uint64
	Current     uint64
	Anon        uint64
	SwapCurrent uint64
	SwapMax     uint64
	HasLimit    bool
	HasSwapMax  bool
}

// HostMemPressure is the admission view: cgroup RAM/swap, not host MemAvailable.
type HostMemPressure struct {
	Cgroup   CgroupMem
	Pressure bool
	Reason   string
}

// ClientMessage is the JSON error body for 503 when we refuse work.
func (p HostMemPressure) ClientMessage() string {
	if p.Reason != "" {
		return p.Reason
	}
	return "host RAM or swap is exhausted; refusing new inference so the server is not OOM-killed"
}

func evaluateHostMemPressure(m CgroupMem, memRatio, swapRatio float64, swapFloor uint64, requireSwap bool) HostMemPressure {
	out := HostMemPressure{Cgroup: m}
	if !m.HasLimit || m.Limit == 0 {
		return out
	}
	used := m.Current
	if m.Anon > used {
		used = m.Anon
	}
	ratio := float64(used) / float64(m.Limit)
	remain := uint64(0)
	if m.Limit > used {
		remain = m.Limit - used
	}

	swapHot := m.SwapCurrent >= swapFloor
	if m.HasSwapMax && m.SwapMax > 0 && swapRatio > 0 {
		if float64(m.SwapCurrent)/float64(m.SwapMax) >= swapRatio {
			swapHot = true
		}
	}

	if ratio >= memRatio && (!requireSwap || swapHot) {
		out.Pressure = true
		out.Reason = fmt.Sprintf(
			"RAM/swap exhausted (anon %s / limit %s, swap %s); refusing new work so the process is not OOM-killed — retry later or free memory",
			format.HumanBytes2(used), format.HumanBytes2(m.Limit), format.HumanBytes2(m.SwapCurrent),
		)
		return out
	}
	if remain < 512*format.MebiByte && m.SwapCurrent > 256*format.MebiByte {
		out.Pressure = true
		out.Reason = fmt.Sprintf(
			"RAM nearly full (anon %s / limit %s, swap %s); refusing new work so the process is not OOM-killed — retry later or free memory",
			format.HumanBytes2(used), format.HumanBytes2(m.Limit), format.HumanBytes2(m.SwapCurrent),
		)
	}
	return out
}
