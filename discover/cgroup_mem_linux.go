//go:build linux

package discover

import (
	"bufio"
	"os"
	"strconv"
	"strings"
)

func readCgroupMem() CgroupMem {
	var m CgroupMem
	if limit, ok := readCgroupMemoryMax(); ok {
		m.Limit = limit
		m.HasLimit = true
	}
	if v, err := getUint64ValueFromFile("/sys/fs/cgroup/memory.current"); err == nil {
		m.Current = v
	}
	if v, err := getUint64ValueFromFile("/sys/fs/cgroup/memory.swap.current"); err == nil {
		m.SwapCurrent = v
	}
	if line, err := readFirstLine("/sys/fs/cgroup/memory.swap.max"); err == nil && line != "max" {
		if v, err := strconv.ParseUint(line, 10, 64); err == nil {
			m.SwapMax = v
			m.HasSwapMax = true
		}
	}
	m.Anon = readCgroupStatAnon()
	if !m.HasLimit {
		if total := procMeminfoKb("MemTotal"); total > 0 {
			m.Limit = total * 1024
			m.HasLimit = true
		}
	}
	if !m.HasSwapMax {
		if total := procMeminfoKb("SwapTotal"); total > 0 {
			m.SwapMax = total * 1024
			m.HasSwapMax = true
		}
	}
	return m
}

func procMeminfoKb(key string) uint64 {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	prefix := key + ":"
	for s.Scan() {
		line := s.Text()
		if strings.HasPrefix(line, prefix) {
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				v, err := strconv.ParseUint(fields[1], 10, 64)
				if err == nil {
					return v
				}
			}
		}
	}
	return 0
}

func readCgroupStatAnon() uint64 {
	f, err := os.Open("/sys/fs/cgroup/memory.stat")
	if err != nil {
		return 0
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	for s.Scan() {
		line := s.Text()
		if strings.HasPrefix(line, "anon ") {
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				v, err := strconv.ParseUint(fields[1], 10, 64)
				if err == nil {
					return v
				}
			}
		}
	}
	return 0
}

// HostMemSnapshot reads cgroup memory and decides whether to refuse inference.
func HostMemSnapshot(memRatio, swapRatio float64, swapFloor uint64) HostMemPressure {
	return evaluateHostMemPressure(readCgroupMem(), memRatio, swapRatio, swapFloor, true)
}

func getCPUMemByCgroups(mem memInfo) memInfo {
	cg := readCgroupMem()
	return applyCgroupMemoryLimit(mem, cg.Limit, cg.HasLimit, cg)
}
