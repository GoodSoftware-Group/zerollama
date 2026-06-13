//go:build darwin && arm64

package discover

import "github.com/ollama/ollama/ml"

// applyMetalUnifiedFreeMemory fills Metal device FreeMemory from the unified host
// pool when bootstrap refresh did not update a device. Why: Metal "VRAM" on Apple
// Silicon is host memory; the ollama-engine discovery subprocess reports process-local
// free bytes that go stale after loads. Scheduler layer fit needs current pool headroom.
func applyMetalUnifiedFreeMemory(devices []ml.DeviceInfo, updated []bool) {
	host, err := GetCPUMem()
	if err != nil || host.FreeMemory == 0 {
		return
	}
	for i := range devices {
		if updated != nil && updated[i] {
			continue
		}
		if devices[i].Library != "Metal" {
			continue
		}
		devices[i].FreeMemory = capMetalUnifiedFree(devices[i].TotalMemory, host.FreeMemory)
		if updated != nil {
			updated[i] = true
		}
	}
}
