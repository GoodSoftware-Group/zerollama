//go:build darwin

package discover

import "os/exec"

func darwinSwapBytes() (total, used uint64) {
	out, err := exec.Command("sysctl", "vm.swapusage").Output()
	if err != nil {
		return 0, 0
	}
	return parseDarwinSwapUsage(string(out))
}

// HostMemSnapshot uses physical RAM + vm.swapusage. Unified memory has no cgroup;
// 88% of RAM is enough to refuse even before swap appears.
func HostMemSnapshot(memRatio, swapRatio float64, swapFloor uint64) HostMemPressure {
	mem, err := GetCPUMem()
	if err != nil || mem.TotalMemory == 0 {
		return HostMemPressure{}
	}
	used := uint64(0)
	if mem.TotalMemory > mem.FreeMemory {
		used = mem.TotalMemory - mem.FreeMemory
	}
	swapTotal, swapUsed := darwinSwapBytes()
	cg := CgroupMem{
		Limit:       mem.TotalMemory,
		HasLimit:    true,
		Current:     used,
		Anon:        used,
		SwapCurrent: swapUsed,
		SwapMax:     swapTotal,
		HasSwapMax:  swapTotal > 0,
	}
	return evaluateHostMemPressure(cg, memRatio, swapRatio, swapFloor, false)
}
