//go:build !linux && !darwin

package discover

func HostMemSnapshot(memRatio, swapRatio float64, swapFloor uint64) HostMemPressure {
	return HostMemPressure{}
}
