package discover

import (
	"context"
	"log/slog"

	"github.com/ollama/ollama/ml"
)

// filterOldCUDADriver drops pre-Volta CUDA GPUs when the installed NVIDIA driver
// is too old to run them safely on cuda_v12 builds. Upstream added this after
// Pascal-era cards started failing at load time with newer llama.cpp CUDA blobs.
func filterOldCUDADriver(_ context.Context, devices []ml.DeviceInfo) []ml.DeviceInfo {
	oldCUDA := func(dev ml.DeviceInfo) bool {
		return dev.Library == "CUDA" && dev.ComputeMajor > 0 && dev.ComputeMajor < 7
	}

	needsCheck := false
	for _, dev := range devices {
		if oldCUDA(dev) {
			needsCheck = true
			break
		}
	}
	if !needsCheck {
		return devices
	}

	driver := nvidiaDriverMajorFromDevices(devices)
	if driver == 0 {
		slog.Warn("could not verify NVIDIA driver compatibility for an older NVIDIA GPU")
		return devices
	}
	if driver >= 570 {
		return devices
	}

	filtered := devices[:0]
	for _, dev := range devices {
		if oldCUDA(dev) {
			slog.Warn("NVIDIA driver too old",
				"device", dev.Description, "compute", dev.Compute(), "driver", driver, "required_driver", "570 or newer")
			continue
		}
		filtered = append(filtered, dev)
	}
	return filtered
}

func nvidiaDriverMajorFromDevices(devices []ml.DeviceInfo) int {
	for _, dev := range devices {
		if dev.Library == "CUDA" && dev.NVIDIADriverMajor > 0 {
			return dev.NVIDIADriverMajor
		}
	}
	return 0
}
