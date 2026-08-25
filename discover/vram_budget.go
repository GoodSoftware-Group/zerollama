package discover

import (
	"log/slog"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/ml"
)

func applyVRAMBudget(devices []ml.DeviceInfo) []ml.DeviceInfo {
	b := envconfig.VRAMBudgetFromEnv()
	if !b.IsSet() || len(devices) == 0 {
		return devices
	}
	for i := range devices {
		if devices[i].TotalMemory == 0 {
			continue
		}
		t, f := b.Apply(devices[i].TotalMemory, devices[i].FreeMemory)
		if t != devices[i].TotalMemory || f != devices[i].FreeMemory {
			slog.Info("applied ZEROLLAMA_VRAM_BUDGET",
				"budget", b.String(),
				"gpu", devices[i].ID,
				"total", t,
				"free", f,
			)
		}
		devices[i].TotalMemory = t
		devices[i].FreeMemory = f
	}
	return devices
}
