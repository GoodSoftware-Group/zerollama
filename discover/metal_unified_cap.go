package discover

// capMetalUnifiedFree clamps host free memory to device total for Metal scheduling.
// Why: unified memory reports one pool; a Metal device's FreeMemory must not exceed
// its TotalMemory or scheduler admission over-estimates headroom per device.
func capMetalUnifiedFree(total, hostFree uint64) uint64 {
	if hostFree == 0 {
		return 0
	}
	free := hostFree
	if total > 0 && free > total {
		free = total
	}
	return free
}
